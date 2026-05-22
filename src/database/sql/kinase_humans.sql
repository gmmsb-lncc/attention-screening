-- psql -U leon -d chembl_35
-- Criando a tabela smile_kinase_human_compounds

CREATE TABLE public.smile_kinase_human_compounds AS
SELECT DISTINCT
    d.chembl_id,                    -- Identificador ChEMBL do composto
    cs.molregno,                   -- Identificador molecular
    t.pref_name AS target_kinase,  -- Nome preferencial da quinase alvo
    cs.canonical_smiles,           -- Estrutura canônica SMILES
    act.standard_value,            -- Valor experimental (IC50, Ki, Kd)
    act.standard_type,             -- Tipo experimental (IC50, Ki, Kd)
    act.pchembl_value,             -- Valor pChEMBL
    d.pref_name AS compound_name,  -- Nome do composto
    t.organism AS organism,        -- Organismo associado à quinase
    csq.sequence AS seq,           -- Sequência de aminoácidos
    csq.component_id AS seq_id     -- Identificador da sequência
FROM
    compound_structures cs
JOIN
    activities act ON cs.molregno = act.molregno
JOIN
    assays a ON act.assay_id = a.assay_id
JOIN
    target_dictionary t ON a.tid = t.tid
LEFT JOIN
    molecule_dictionary d ON cs.molregno = d.molregno
LEFT JOIN
    target_components tc ON t.tid = tc.tid       -- Associa alvos aos componentes
LEFT JOIN
    component_sequences csq ON tc.component_id = csq.component_id -- Junta com as sequências
WHERE
    t.pref_name LIKE '%kinase%' AND      -- Filtra apenas alvos relacionados a quinases
    cs.canonical_smiles IS NOT NULL AND -- Garante que SMILES não seja nulo
    act.standard_type IN ('IC50', 'Ki', 'Kd') AND -- Filtra tipos experimentais relevantes
    act.standard_value IS NOT NULL AND  -- Garante que os valores sejam válidos
    act.standard_units = 'nM' AND       -- Garante a unidade padrão
    (act.data_validity_comment IS NULL OR act.data_validity_comment = 'Manually validated') AND -- Apenas dados validados
    t.organism = 'Homo sapiens';        -- Filtra apenas quinases humanas

-- Exporta os dados para um arquivo TSV
\COPY public.smile_kinase_human_compounds TO '${PROJECT_ROOT}/src/database/kinase_human_compounds.tsv' WITH (FORMAT csv, HEADER, DELIMITER E'\t');