
SELECT TOP 780000 sentence FROM Sentences
WHERE token_count > 2 and token_count <=100 
ORDER BY id
