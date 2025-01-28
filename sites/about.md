# 🧑🏼‍💻 Über ParteiPapagei
_Demokratie ist der Wettbewerb politischer Ideen. Die Parteien stellen ihre Ideen in ihrem Programm vor und am Wahltag entscheiden die Wähler, welche Idee sie am überzeugendensten finden. Mit ParteiPapagei wollen wir diese Ideen dem Wähler leichter zugänglich machen._


ParteiPapagei wurde von Tim Baumgärtner entwickelt und ist ein Open-Source-Projekt. Der Quellcode ist auf [GitHub](https://github.com/timbmg/ParteiPapagei) verfügbar. Es wird keine Garantie für die Richtigkeit der generierten Inhalte übernommen. Die Informationen sind nicht verbindlich und dienen nur zur Bildung und Unterhaltung. Die Verwendung der Informationen erfolgt auf eigene Gefahr. Siehe auch den [Disclaimer](/disclaimer) und die [Datenschutzbestimmungen](/data-protection).

Für Fragen und Anregungen können Sie mich unter [baumgaertner.t@gmail.com](mailto:baumgaertner.t@gmail.com) erreichen.

Für Feature Requests und Bug Reports können Sie gerne ein [Issue](https://github.com/timbmg/ParteiPapagei/issues) auf GitHub erstellen.

🤲 Falls Du ParteiPapagei hilfreich findest und einen Teil zu den laufenden Kosten beitragen kannst, unterstüze mich gerne auf [Ko-fi](https://ko-fi.com/timbmg).

## Technologie
ParteiPapagei basiert auf Retrieval Augmented Generation (RAG) [[1](#refRAG)]. Dabei wird eine Suche mit einem Large Language Model (LLM) verknüpft. Im Fall von ParteiPapagei werden zunächst relevante Passagen aus dem Wahlprogramm gesucht, basierend auf der Eingabe des Nutzers. Abschließend werden die relevanten Passagen genutzt, um eine Antwort zu generieren.

![RAG](./rag.png)

### Preprocessing
Zunächst wurden die PDFs der Wahlprogramme halb-automatisch in Markdown überführt. Anschließend wird der Text in Passagen unterteilt, falls diese noch zu lang sind, werden sie wiederum aufgeteilt ("Chunking"). Zusätzlich zum Text wird außerdem die Überschrift der Passagen gespeichert, um den Kontext der Passage später besser zu verstehen.

### Textanalyse
Die Passagen des Preprocessings werden anschließend einer Schlagwort- und Semantischen Analyse unterzogen. Dabei werden die Schlagwörter der Passagen extrahiert und gespeichert. Zusätzlich wird eine semantische Repräsentation der Passagen erstellt, um Passagen nicht nur anhand von Schlagwörtern, sondern auch anhand des Inhalts zu finden.

### Fragen Verarbeitung
Basierend auf der Frage des Nutzers werden zunächst ähnliche Fragen generiert. Dies dient dazu, mehr relevante Schlagwörter oder andere Informationen aus dem Wahlprogramm zu finden, die so nicht direkt in der Frage enthalten sind. Anschließend werden die Fragen derselben Textanalyse unterzogen wie die Passagen.

### Suche
Als Suche nutzt ParteiPapagei dabei zum einen eine Schlagwortsuche (BM25 [[2](#refBM25)]), zum anderen eine semantische Suche basierend auf einem Dense Retriever. Die semantische Suche wird durch Googles Gemini Model (`models/text-embedding-004`) [[3](#refGecko)] realisiert. 
Nachdem sowohl die Wahlprogramme, als auch die Suchanfrage in eine semantische Repräsentation umgewandelt wurden, wird die Ähnlichkeit zwischen den beiden Repräsentationen berechnet. Am Ende werden die Passagen aus den Wahlprogrammen zurückgegeben, die die höchste Ähnlichkeit zur Suchanfrage aufweisen. Die Ergebnisse der beiden Repräsentationen werden dabei fusioniert [[4](#refRR)].


### Antwortgenerierung
Die Passagen aus dem Wahlprogramm, die durch die Suche gefunden wurden, werden dann genutzt, um eine Antwort zu generieren. Dabei wird ein Large Language Model (LLM) genutzt. In ParteiPapagei wird Google's Gemini Model (`models/gemini-1.5-flash-002`) [[5](#refGemini)] verwendet. Dabei handelt es sich um ein relativ kleines Modell, das zwar nicht so leistungsstark wie größere Modelle ist, aber relativ günstig genutzt werden kann. Außer der Generierung der Antwort wird das Modell instruiert, die Passagen aus dem Wahlprogramm zu zitieren. Diese werden dann als Links in der Antwort angezeigt, sodass der Nutzer die Quelle überprüfen kann.

### Einschränkungen
- ParteiPapagei besitzt kein "Gedächtnis" und berücksichtigt Kontext (also die vorherigen Nachrichten) zur Beantwortung der Fragen nicht. D.h. jede Frage wird unabhängig beantwortet und Folgefragen können nicht auf vorherige Antworten Bezug nehmen.
- ParteiPapagei kann nur auf die Informationen in den Wahlprogrammen zugreifen. Es kann keine aktuellen Informationen oder Meinungen zu politischen Themen geben.
- Erfahrungsgemäß funktionieren sehr allgemein gehaltene Fragen weniger gut, da die Suche nach relevanten Passagen schwieriger ist.
- ParteiPapagei benutzt LLMs für die Suche und Generierung von Antworten. Diese Modelle sind nicht immer neutral und können bestimmte Bias enthalten [[6](refBias)].

## Referenzen

<a name="refRAG"></a>[1] Lewis, Patrick, et al. ["Retrieval-augmented generation for knowledge-intensive nlp tasks."](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) Advances in Neural Information Processing Systems 33 (2020): 9459-9474.

<a name="refBM25"></a>[2] Robertson, Stephen, and Hugo Zaragoza. ["The probabilistic relevance framework: BM25 and beyond."](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) Foundations and Trends® in Information Retrieval 3.4 (2009): 333-389.

<a name="refGecko"></a>[3] Lee, Jinhyuk, et al. ["Gecko: Versatile text embeddings distilled from large language models."](https://arxiv.org/pdf/2403.20327) arXiv preprint arXiv:2403.20327 (2024).

<a name="refRR"></a>[4] Cormack, Gordon V., Charles LA Clarke, and Stefan Buettcher. ["Reciprocal rank fusion outperforms condorcet and individual rank learning methods."](https://dl.acm.org/doi/pdf/10.1145/1571941.1572114) Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval. 2009.

<a name="refGemini"></a>[5] Team, Gemini, et al. ["Gemini: a family of highly capable multimodal models."](https://arxiv.org/pdf/2312.11805) arXiv preprint arXiv:2312.11805 (2023).

<a name="refBias"></a>[6] Yejin Bang, Delong Chen, Nayeon Lee, Pascale Fung. ["Measuring Political Bias in Large Language Models: What Is Said and How It Is Said"](https://aclanthology.org/2024.acl-long.600/) Annual Meeting of the Association for Computational Linguistics. 2024.
