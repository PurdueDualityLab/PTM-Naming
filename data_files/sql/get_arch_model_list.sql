SELECT model.context_id, architecture.name
FROM 
    model,
    model_to_architecture, 
    architecture
WHERE architecture.id = model_to_architecture.architecture_id
    AND model.id = model_to_architecture.model_id