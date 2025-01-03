import xxhash

def create_anchor_from_text(text: str | None) -> str:
    # based on https://github.com/streamlit/streamlit/blob/833efa9fe408c692906bd07b201b5e715bcceae2/frontend/lib/src/components/shared/StreamlitMarkdown/StreamlitMarkdown.tsx#L121-L137
    new_anchor = ""
    
    # Check if the text is valid ASCII characters
    is_ascii = text and all(ord(c) < 128 for c in text)
    
    if is_ascii and text:
        new_anchor = text.lower().replace(" ", "-")
    elif text:
        # If the text is not valid ASCII, use a hash of the text
        new_anchor = xxhash.xxh32(text, seed=0xabcd).hexdigest()[:16]
    
    return new_anchor
