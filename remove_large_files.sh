   #!/bin/bash
   git filter-branch --force --index-filter \
   "git rm --cached --ignore-unmatch \
   before/formerNegativeSampling/dense_embedding.bin \
   before/formerNegativeSampling/negative_samples.pkl \
   before/formerNegativeSampling/p_encoder/model.safetensors \
   before/formerNegativeSampling/q_encoder/model.safetensors \
   formerNegativeSampling/dense_embedding.bin \
   formerNegativeSampling/negative_samples.pkl \
   formerNegativeSampling/p_encoder/model.safetensors \
   formerNegativeSampling/q_encoder/model.safetensors" \
   --prune-empty --tag-name-filter cat -- --all