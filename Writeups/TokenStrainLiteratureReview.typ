One of the factors we assumed to have an effect on the model's acquiring of knowledge is the interference of tokens that are shared between entities. We call this the "token strain" effect. The idea is that multiple contexts in which a token appears may compete on the model's attention or 'memory' for that token, degrading the overall performance of the model on entities that share tokens. Chang et al. 2024 @chang2024characterizing names this phenomenon as "Target token contextual diversity" and measures it as "number of unique
tokens that appear within 30 tokens of the
target token in the pre-training dataset". They find that tokens that appear in more diverse contexts receive higher forgetting score (measured as surprisal) with an effect that is extremely small but statistically significant. 


Utillizing the Wikipedia QID's approach taken in LMEnt, we can extract actual contextual diversity of tokens, as well as the number of chunks in which a token appears out of the target entity's context.






#bibliography("project.bib")
