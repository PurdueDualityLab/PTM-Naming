SELECT model.context_id, hf_commit.created_at
FROM hf_commit, model
WHERE hf_commit.model_id = model.id