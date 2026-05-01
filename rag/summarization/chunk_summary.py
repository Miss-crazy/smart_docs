from langchain.chains.summarize import load_summarize_chain 

def summarize_chunks(chunks:list[str], llm)-> str:
    chain = load_summarize_chain(llm, chain_type="stuff")
    return chain.run(chunks)

