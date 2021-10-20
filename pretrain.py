import yake

class Yake_KE():
    def __init__(self):
        self.kw_extractor = yake.KeywordExtractor()
    
    def extract_keywords(self, text: str, stemming = False) -> list:
        weighted_keywords = self.kw_extractor.extract_keywords(text)
        unweighted_keywords = [key[0] for key in weighted_keywords]
        return unweighted_keywords
    
    def extract_keywords_with_weights(self, text: str, stemming = False) -> list:
        weighted_keywords = self.kw_extractor.extract_keywords(text)
        return weighted_keywords