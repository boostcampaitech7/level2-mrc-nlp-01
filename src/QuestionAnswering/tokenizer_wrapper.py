from transformers import EvalPrediction

class QuestionAnsweringTokenizerWrapper:
    def __init__(self, tokenizer, config, column_name_dict = {"question": "question", "context": "context", "answers": "answers"}):
        self.tokenizer = tokenizer
        
        # 행 이름을 기억합니다.
        self.question_column = column_name_dict.get("question", "question")
        self.context_column = column_name_dict.get("context", "context")
        self.answer_column = column_name_dict.get("answers", "answers")
        
        self.max_seq_length = config.dataQA.tokenizer.max_seq_length(384)
        self.doc_stride = config.dataQA.tokenizer.doc_stride(128)
        self.pad_to_max_length = config.dataQA.tokenizer.pad_to_max_length(True)
        
    def tokenize(self, examples):
        pad_on_right = self.tokenizer.padding_side == "right"
        return self.tokenizer(
            examples[self.question_column if pad_on_right else self.context_column],
            examples[self.context_column if pad_on_right else self.question_column],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if self.pad_to_max_length else False,
        )
    
    def encode_train(self, examples):
        """
        Train dataset을 다음과 같이 인코딩합니다.
        1.  tokenizer를 이용하여 tokenization을 진행합니다.
            이때 context가 너무 길면 여러 개의 context로 나누어지게 됩니다.
            따라서, 동일한 query에 대하여 여러 개의 example이 생성됩니다.
            ex) [Q, [C1, C2, C3, ...]] -> [Q, [C1]], [Q, [C2]], [Q, [C3]], ...
        2.  각 example에 대하여 context에 포함된 정답 token span을 찾아서
            start_positions, end_positions에 저장합니다.
            정답이 없는 경우 cls token을 정답으로 설정합니다.
            
        Args:
            examples : encode가 필요한 dataset

        Returns:
            tokenized_example : encode된 dataset
        """
        tokenized_example = self.tokenize(examples)
        
        # return_overflowing_tokens=True로 설정하면 overflow_to_sample_mapping이 생성됩니다.
        sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")
        # 하나의 context가 잘려서 여러 개로 나누어지면, 
        # sample_mapping을 통해 원래 몇 번째의 context를 가리키는지 알 수 있습니다.
        
        # return_offsets_mapping=True로 설정하면 offset_mapping이 생성됩니다.
        offset_mapping = tokenized_example.pop("offset_mapping")
        # 각 토큰이 원래 question+context의 어디 index에 해당하는지 알 수 있습니다.
        
        tokenized_example["start_positions"] = []
        tokenized_example["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping): # 쪼개진 후의 examples에 대한 반복문
            # 각 example에 대하여
            # 어떤 token span이 정답에 해당하는지 구하는 코드입니다.
            # 결과는 start_positions, end_positions에 저장됩니다.
            # 정답이 없는 경우 cls token을 정답으로 설정합니다.
            input_ids = tokenized_example["input_ids"][i]
            cls_idx = input_ids.index(self.tokenizer.cls_token_id)
            
            sample_idx = sample_mapping[i]
            answer_idx = examples[self.answer_column][sample_idx]
            
            if len(answer_idx["answer_start"]) == 0:
                # 답이 없는 경우
                tokenized_example["start_positions"].append(cls_idx)
                tokenized_example["end_positions"].append(cls_idx)
            else: 
                # 답이 있는 경우
                start_char = answer_idx["answer_start"][0]
                end_char = start_char + len(answer_idx["text"][0])
                # (start_char, end_char) -> context에서 정답의 시작과 끝 index
                
                sequence_ids = tokenized_example.sequence_ids(i) 
                # tokenized_example.sequence_ids(i) -> i번째 example의 sequence_id 반환
                # sequence_id는 question(0-left) + context(1-right)로 구성되어 있음
                
                token_start_idx = 0
                pad_on_right = self.tokenizer.padding_side == "right"
                while sequence_ids[token_start_idx] != (1 if pad_on_right else 0):
                    token_start_idx += 1
                
                token_end_idx = len(input_ids) - 1
                while sequence_ids[token_end_idx] != (1 if pad_on_right else 0):
                    token_end_idx -= 1
                    
                # token_start_idx, token_end_idx -> context에서의 시작과 끝 index (토큰)
                
                if (offsets[token_start_idx][0] > start_char
                   or offsets[token_end_idx][1] < end_char):
                    # 답이 context를 벗어난 경우
                    tokenized_example["start_positions"].append(cls_idx)
                    tokenized_example["end_positions"].append(cls_idx)
                else:
                    
                    while (
                        token_start_idx < len(offsets)
                        and offsets[token_start_idx][0] <= start_char # 같아도 +1 하므로 마지막에 -1 해주어야 한다.
                    ):
                        token_start_idx += 1
                    token_start_idx -= 1
                    
                    while offsets[token_end_idx][1] >= end_char: # 같아도 -1 하므로 마지막에 +1 해주어야 한다.
                        token_end_idx -= 1
                    token_end_idx += 1
                    
                    tokenized_example["start_positions"].append(token_start_idx)
                    tokenized_example["end_positions"].append(token_end_idx)
        return tokenized_example
    
    def encode_valid(self, examples):
        """
        train dataset과 비슷하게 validation dataset에 대한 encode를 진행합니다.
        1. validation dataset의 decode를 위해 example_id를 저장합니다.
        2. offset_mapping을 보고 context의 일부가 아니라면 None으로 설정합니다.

        Args:
            examples : encode가 필요한 dataset
            
        Returns:
            tokenized_example : encode된 dataset
        """
        tokenized_example = self.tokenize(examples)
        
        sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")
        
        tokenized_example["example_id"] = []
        for i, sample_idx in enumerate(sample_mapping): # tokenize 이후의 쪼개진 example에 대하여
            original_id = examples["id"][sample_idx]
            tokenized_example["example_id"].append(original_id)
            
            sequence = tokenized_example.sequence_ids(i)
            context_idx = 1 if self.tokenizer.padding_side == "right" else 0
            tokenized_example["offset_mapping"][i] = [
                (o if sequence[k] == context_idx else None)
                for k, o in enumerate(tokenized_example["offset_mapping"][i])
            ]
        
        return tokenized_example
    
    def decode(self, examples, predictions, training_args):
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            return EvalPrediction(
                predictions=formatted_predictions,
                label_ids=[{"id": ex["id"], "answers": ex[self.answer_column]} for ex in examples],
            )


class Seq2SeqLMTokenizerWrapper:
    def __init__(self, tokenizer, config, column_name_dict={"question": "question", "context": "context", "answers": "answers"}):
        self.tokenizer = tokenizer
        
        # 컬럼 이름을 저장합니다.
        self.question_column = column_name_dict.get("question", "question")
        self.context_column = column_name_dict.get("context", "context")
        self.answer_column = column_name_dict.get("answers", "answers")
        
        # 토크나이저 설정
        self.max_seq_length = config.dataQA.tokenizer.max_seq_length(512)
        self.pad_to_max_length = config.dataQA.tokenizer.pad_to_max_length(True)
        
    def tokenize(self, examples):
        # 질문과 문맥을 하나의 시퀀스로 결합하여 토크나이징
        inputs = [
            f"question: {q} context: {c}" 
            for q, c in zip(examples[self.question_column], examples[self.context_column])
        ]
        
        # 모델 입력을 위한 토크나이징
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length" if self.pad_to_max_length else False
        )
        
        return model_inputs

    def encode_train(self, examples):
        """
        학습 데이터를 인코딩합니다. 질문과 문맥을 결합한 후 답변을 레이블로 설정합니다.
        """
        tokenized_example = self.tokenize(examples)

        # 답변을 레이블로 설정
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [answer["text"][0] for answer in examples[self.answer_column]], 
                max_length=self.max_seq_length,
                truncation=True,
                padding="max_length" if self.pad_to_max_length else False
            )
        
        tokenized_example["labels"] = labels["input_ids"]
        return tokenized_example

    def encode_valid(self, examples):
        """
        검증 데이터를 인코딩합니다. 학습과 동일하지만 example_id를 추가로 저장합니다.
        """
        tokenized_example = self.tokenize(examples)

        # 검증을 위한 example_id 저장
        tokenized_example["example_id"] = examples["id"]
        
        # 검증을 위한 레이블 설정
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [answer["text"][0] for answer in examples[self.answer_column]],
                max_length=self.max_seq_length,
                truncation=True,
                padding="max_length" if self.pad_to_max_length else False
            )

        tokenized_example["labels"] = labels["input_ids"]
        return tokenized_example

    def decode(self, examples, predictions, training_args):
        """
        예측 결과를 디코딩하고, 평가에 적합한 포맷으로 반환합니다.
        """
        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        formatted_predictions = [{"id": ex["id"], "prediction_text": pred} for ex, pred in zip(examples, decoded_predictions)]
        
        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            return EvalPrediction(
                predictions=formatted_predictions,
                label_ids=[{"id": ex["id"], "answers": ex[self.answer_column]} for ex in examples],
            )
