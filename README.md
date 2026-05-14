#Do LLMs translate the same way as human experts?
**Motivation**
I loved reading the modern Spanish classic, "Love in the time of Cholera" by Gabriel Garcia Marquez - one of the great novels of the 20th century.  The English translation was done by Edith Grossman, who describes translating previous literature in one language into another as, "Translating is always a struggle, regardless of the author you're translating. You have to hear the original voice in a profound way, and then find the voice in English that best reflects that original. It's always difficult, challenging and immensely enjoyable.".  I found myself re-reading passages that took my breath away and wondered how a translation could have this effect and if I would the original Spanish text would have had the same effect or greater.  While that is a complex human phenomena to uncover, I also wondered if LLMs can translate that beautifully.  So, I decided on a small pilot study.  I took one of my favorite passages from the English translation, then I took the original Spanish translation of that passage and had Gemini and ChatGPT-5 provide their translations. The analysis code that follows compares the human translation to the two LLM translation.  What was discovered was while the overall cosine similarity across all layers was very similar to the human's translation, the two LLMs' translation was more similar to each other than they were to the Edith Grossman's.  Findings include consistent geometric separation between human and LLM translations 
across all 12 layers, maximum discriminability at layer 9, and token-level evidence that Grossman's translation is characterized by physical/sensory vocabulary while LLM translations favor abstract and explanatory word choices. 

**Approach**
  - Model and tools
  - Translations used
  - Why this passage
**Models and Tools**



