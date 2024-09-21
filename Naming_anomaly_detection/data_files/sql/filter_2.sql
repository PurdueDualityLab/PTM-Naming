-- WITH filter2 AS (
--     SELECT architecture.id, COUNT(architecture.id) AS model_count
--     FROM architecture
--     JOIN model_to_architecture ON architecture.id = model_to_architecture.architecture_id
--     JOIN model ON model.id = model_to_architecture.model_id
--     GROUP BY architecture.id
--     HAVING COUNT(architecture.id) >= 20
-- ),
WITH filter2 AS (
    SELECT architecture.id AS architecture_id, COUNT(model.id) AS model_count
    FROM architecture
    JOIN model_to_architecture ON architecture.id = model_to_architecture.architecture_id
    JOIN model ON model.id = model_to_architecture.model_id
    GROUP BY architecture.id
    HAVING COUNT(model.id) >= 20
),

filter1 AS (
    SELECT model.context_id, architecture.name, model.downloads
    FROM model
    JOIN model_to_architecture ON model.id = model_to_architecture.model_id
    JOIN architecture ON architecture.id = model_to_architecture.architecture_id
    -- WHERE model.downloads >= 5
    WHERE architecture.id IN (SELECT architecture_id FROM filter2)
    ORDER BY architecture.name ASC, model.downloads DESC
)

SELECT *
FROM filter1;
