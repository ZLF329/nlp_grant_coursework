import textstat
import re

class PlainEnglishEvaluator:
    def __init__(self):
        self.target_grade_range = (9, 12)

    def get_fk_grade(self, text: str) -> float:
        """Calculates the Flesch-Kincaid Grade Level."""
        if not text.strip():
            return 0.0
        return textstat.flesch_kincaid_grade(text)
    
    def get_jargon_list(self, text: str) -> list:
        """
        Extracts a list of words that are considered 'difficult' 
        based on standard easy-word dictionaries.
        """
        if not text.strip():
            return []
        return textstat.difficult_words_list(text)
    
    def get_jargon_density(self, jargon_list: list, total_words: int) -> float:
        """Calculates the percentage of difficult words in the text."""
        if total_words == 0:
            return 0.0
        return (len(jargon_list) / total_words) * 100
    
    def analyze_text(self, text: str, total_words: int):
        fk_score = self.get_fk_grade(text)
        jargon_list = self.get_jargon_list(text)
        jargon_density = self.get_jargon_density(jargon_list, total_words)

        return {
            "flesch_kincaid_grade": round(fk_score, 2),
            "jargon_density": f"{round(jargon_density, 2)}%",
            "jargon_count": len(jargon_list),
            "is_grade_level_ok": self.target_grade_range[0] <= fk_score <= self.target_grade_range[1]
        }