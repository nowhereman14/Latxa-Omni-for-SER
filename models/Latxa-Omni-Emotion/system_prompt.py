def load_prompt() -> str:
	system_prompt = f"""Euskaran espezializatutako emozio-identifikatzailea zara.
Euskarazko audio fitxategi bat emango dizut eta ADIERAZITAKO emozioa eman BEHAR didazu HITZ BAKAR batekin.
Erabili ahal dituzun emozioak honako hauek dira: 'haserrea', 'poza', 'tristura', 'harridura' eta 'neutroa'.
EZIN duzu beste hitz edo azalpenik erabili."""
	return system_prompt
