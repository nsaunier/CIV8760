-- Exemples
-- Compter le nombre de lignes dans la table perception
SELECT count(*) FROM perception;
-- Compter le nombre de transactions par jour
SELECT date, count(*) as achalandage FROM perception GROUP BY date;
-- Compter le nombre de transactions par type de titre, puis par type d'abonnement (exercice)
SELECT titre, count(*) FROM perception GROUP BY titre;
-- Identifier tous les titre, carte_type, abonnement, tarif (exercice)
SELECT DISTINCT titre FROM perception;
SELECT DISTINCT titre, carte_type FROM perception;
-- Trouver la station de métro la plus achalandée, par type de carte et d'abonnement (mêmes questions pour les bus)
SELECT station, count(*) as achalandage FROM perception
WHERE station <> 0
GROUP BY station
ORDER BY achalandage DESC;
SELECT station, carte_type, count(*) as achalandage FROM perception
WHERE station <> 0
GROUP BY station, carte_type
ORDER BY achalandage DESC;
-- Identifier les usagers les plus actifs
SELECT carte_num, count(*) as n FROM perception GROUP BY carte_num ORDER BY n DESC limit 10;
-- Identifier la distribution du nombre de trajets faits par les usagers par type de jour
SELECT strftime("%w", date) as jour, count(*) FROM perception GROUP BY jour;
