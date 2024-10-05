import yaml

class Config:
    def __init__(self, config_dict=None, path='./config.yaml'):
        if path is None:
            return
        if config_dict is None:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        self._make(config_dict)
    
    def _make(self, configs):
        if not isinstance(configs, dict):
            if configs is not None:
                self.atom = configs
        else:
            for key, value in configs.items():
                if isinstance(value, list):
                    v_config = []
                    for v in value:
                        v_ele = Config(path=None)
                        v_ele._make(v)
                        v_config.append(v_ele)
                else:
                    v_config = Config(path=None)
                    v_config._make(value)
                setattr(self, key, v_config)
    
    def __repr__(self):
        return yaml.dump(self._to_dict(), indent=4)  # 들여쓰기 4칸으로 JSON 형식 출력

    def _to_dict(self):
        # 객체를 딕셔너리로 변환하는 메서드
        if self.atom is not None:
            return self.atom
        
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value._to_dict()  # 재귀적으로 딕셔너리로 변환
            elif isinstance(value, list):
                result[key] = [v._to_dict() if isinstance(v, Config) else v for v in value]
            else:
                result[key] = value
        return result

    def __getattr__(self, name):
        if name == 'atom':
            return None
        return Config(path=None)
    
    def __call__(self, default_value=None):
        if self.atom is not None:
            return self.atom
        
        if len(self.__dict__) > 0:
            return self._to_dict()
        
        if default_value is not None:
            return default_value
        
        return None