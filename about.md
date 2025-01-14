# Über ChatBTW
ChatBTW wurde von Tim Baumgärtner entwickelt und ist ein Open-Source-Projekt. Der Quellcode ist auf [GitHub](https://github.com/timbmg/chatbtw) verfügbar. Es wird keine Garantie für die Richtigkeit der generierten Inhalte übernommen. Die Informationen sind nicht verbindlich und dienen nur zur Bildung und Unterhaltung. Die Verwendung der Informationen erfolgt auf eigene Gefahr. Siehe auch den [Disclaimer](/disclaimer) und die [Datenschutzbestimmungen](/data_protection).

Für Fragen und Anregungen können Sie mich unter [baumgaertner.t@gmail.com](mailto:baumgaertner.t@gmail.com) erreichen.

Für Feature Requests und Bug Reports können Sie gerne ein [Issue](https://github.com/timbmg/chatbtw/issues) auf GitHub erstellen.

## Technologie
ChatBTW basiert auf Retrieval Augmented Generation (RAG) [[1](#refRAG)]. Dabei wird eine Suche mit einem Large Lange Model (LLM) verknüpft. Im Fall von ChatBTW werden zunächst relevante Passagen aus dem Wahlprogramm gesucht, basierend auf der Eingabe des Nutzers. Abschließend werden die relevanten Passagen genutzt, um eine Antwort zu generieren.

### Suche
Als Suche nutzt ChatBTW dabei zum einen eine Schlagwortsuche (BM25 [[2](#refBM25)]), zum anderen eine semantische Suche basierend auf einem Dense Retriever. Die semantische Suche wird durch Googles Gemini Model (`models/text-embedding-004`) [[3](#refGecko)] realisiert. Außerdem werden ähnliche Suchanfragen generiert, um die Suche zu verfeinern (z.B. um Synonyme zu finden). Letztlich wird eine Kombination aller gefundenen Ergebnisse genutzt, um die relevanten Passagen zu finden [[4](#refRR)].

### Generierung
Die Passagen aus dem Wahlprogramm, die durch die Suche gefunden wurden, werden dann genutzt, um eine Antwort zu generieren. Dabei wird ein Large Language Model (LLM) genutzt. In ChatBTW wird Google's Gemini Model (`models/gemini-1.5-flash-002`) [[5](#refGemini)] verwendet. Dabei handelt es sich vermutlich um ein relativ kleines Modell, das zwar nicht so leistungsstark wie größere Modelle ist, aber in gewissem Maße kostenlos genutzt werden kann. Außer der Generierung der Antwort wird das Modell instruiert, die Passagen aus dem Wahlprogramm zu zitieren. Diese werden dann als Links in der Antwort angezeigt, sodass der Nutzer die Quelle überprüfen kann.

### Einschränkungen
- ChatBTW besitzt kein "Gedächtnis" und berücksichtigt keinen Kontext (also die vorherigen Nachrichten).
- Erfahrungsgemäß funktionieren sehr allgemein gehaltene Fragen weniger gut.

# Referenzen

<a name="refRAG"></a>[1] Lewis, Patrick, et al. ["Retrieval-augmented generation for knowledge-intensive nlp tasks."](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) Advances in Neural Information Processing Systems 33 (2020): 9459-9474.

<a name="refBM25"></a>[2] Robertson, Stephen, and Hugo Zaragoza. ["The probabilistic relevance framework: BM25 and beyond."](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) Foundations and Trends® in Information Retrieval 3.4 (2009): 333-389.

<a name="refGecko"></a>[3] Lee, Jinhyuk, et al. ["Gecko: Versatile text embeddings distilled from large language models."](https://arxiv.org/pdf/2403.20327) arXiv preprint arXiv:2403.20327 (2024).

<a name="refRR"></a>[4] Cormack, Gordon V., Charles LA Clarke, and Stefan Buettcher. ["Reciprocal rank fusion outperforms condorcet and individual rank learning methods."](https://dl.acm.org/doi/pdf/10.1145/1571941.1572114) Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval. 2009.

<a name="refGemini"></a>[5] Team, Gemini, et al. ["Gemini: a family of highly capable multimodal models."](https://arxiv.org/pdf/2312.11805) arXiv preprint arXiv:2312.11805 (2023).
