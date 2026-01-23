-- 1. Nettoyage des tables existantes (Noms synchronisés)
DROP TABLE IF EXISTS rh_general CASCADE;
DROP TABLE IF EXISTS evaluations_employee CASCADE;
DROP TABLE IF EXISTS sondage_employee CASCADE;

-- 2. Création des tables
CREATE TABLE rh_general (
    id_employee INT PRIMARY KEY,
    age INT,
    genre VARCHAR(10),
    revenu_mensuel INT,
    statut_marital VARCHAR(50),
    departement VARCHAR(100),
    poste VARCHAR(100),
    nombre_experiences_precedentes INT,
    nombre_heures_travailless INT,
    annee_experience_totale INT,
    annees_dans_l_entreprise INT,
    annees_dans_le_poste_actuel INT
);

CREATE TABLE evaluations_employee (
    satisfaction_employee_environnement INT,
    note_evaluation_precedente INT,
    niveau_hierarchique_poste INT,
    satisfaction_employee_nature_travail INT,
    satisfaction_employee_equipe INT,
    satisfaction_employee_equilibre_pro_perso INT,
    eval_number VARCHAR(50) PRIMARY KEY, 
    note_evaluation_actuelle INT,
    heure_supplementaires VARCHAR(5),   
    augementation_salaire_precedente VARCHAR(10) 
);

CREATE TABLE sondage_employee (
    a_quitte_l_entreprise VARCHAR(5),               
    nombre_participation_pee INT,
    nb_formations_suivies INT,
    nombre_employee_sous_responsabilite INT,
    code_sondage VARCHAR(50) PRIMARY KEY,           
    distance_domicile_travail INT,
    niveau_education INT,
    domaine_etude VARCHAR(100),
    ayant_enfants VARCHAR(5),                       
    frequence_deplacement VARCHAR(50),              
    annees_depuis_la_derniere_promotion INT,
    annes_sous_responsable_actuel INT
);

-- 3. Bloc unique de nettoyage et d'analyse (WITH)
WITH 
-- Nettoyage RH
cleaned_rh AS (
    SELECT * FROM rh_general
),

-- Nettoyage Evaluations
cleaned_eval AS (
    SELECT 
        REPLACE(eval_number, 'E_', '')::INT AS id_employee,
        note_evaluation_actuelle,
        CASE WHEN heure_supplementaires = 'Oui' THEN TRUE ELSE FALSE END AS a_fait_heures_sup,
        REPLACE(augementation_salaire_precedente, ' %', '')::INT AS pct_augmentation_salaire
    FROM evaluations_employee
),

-- Nettoyage Sondage
cleaned_survey AS (
    SELECT 
        code_sondage::INT AS id_employee,
        a_quitte_l_entreprise,
        distance_domicile_travail,
        nb_formations_suivies
    FROM sondage_employee
),

-- Jointure Globale (Ici on utilise cleaned_rh et non cleaned_general)
global_employee_data AS (
    SELECT 
        r.*,
        e.note_evaluation_actuelle,
        e.pct_augmentation_salaire,
        e.a_fait_heures_sup,
        s.a_quitte_l_entreprise,
        s.distance_domicile_travail,
        s.nb_formations_suivies
    FROM cleaned_rh r
    LEFT JOIN cleaned_eval e ON r.id_employee = e.id_employee
    LEFT JOIN cleaned_survey s ON r.id_employee = s.id_employee
)

-- Requête finale d'analyse
SELECT 
    a_quitte_l_entreprise AS statut_employe,
    COUNT(*) AS nombre_employes,
    ROUND(AVG(age), 1) AS age_moyen,
    ROUND(AVG(revenu_mensuel), 0) AS salaire_moyen,
    ROUND(AVG(distance_domicile_travail), 1) AS distance_moyenne_km,
    ROUND(AVG(pct_augmentation_salaire), 2) AS augmentation_moyenne_pct
FROM global_employee_data
GROUP BY a_quitte_l_entreprise;