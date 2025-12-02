from typing import List, Dict
from main import get_vectorstore

def keyword_search(query: str, file_name: str = None, limit: int = 7) -> List[Dict]:
    """Perform keyword search on documents"""
    try:
        vs = get_vectorstore()
        if vs is None:
            return []
        
        # Get all documents (or filtered by file_name)
        all_docs = vs.similarity_search(
            query="",
            k=1000,
            expr=f'file_name == "{file_name}"' if file_name else None
        )
        
        # Filter by keyword match
        query_lower = query.lower()
        results = []
        
        for doc in all_docs:
            if query_lower in doc.page_content.lower():
                results.append({
                    "content": doc.page_content,
                    "file_name": doc.metadata.get("file_name", ""),
                    "score": 1.0,
                    "search_type": "keyword"
                })
                if len(results) >= limit:
                    break
        
        print(f"Keyword search found {len(results)} results")
        return results
    
    except Exception as e:
        print(f"Error in keyword search: {e}")
        return []

def vector_search(query: str, file_name: str = None, limit: int = 7) -> List[Dict]:
    """Perform semantic vector search"""
    try:
        vs = get_vectorstore()
        if vs is None:
            return []
        
        search_kwargs = {"k": limit}
        if file_name:
            search_kwargs["expr"] = f'file_name == "{file_name}"'
        
        docs = vs.similarity_search_with_score(query, **search_kwargs)
        
        results = []
        for doc, score in docs:
            results.append({
                "content": doc.page_content,
                "file_name": doc.metadata.get("file_name", ""),
                "score": float(score),
                "search_type": "vector"
            })
        
        print(f"Vector search found {len(results)} results")
        return results
    
    except Exception as e:
        print(f"Error in vector search: {e}")
        return []

def hybrid_search(
    vector_results: List[Dict],
    keyword_results: List[Dict],
    alpha: float = 0.7,
    limit: int = 7
) -> List[Dict]:
    """Combine vector and keyword results with weighted scoring"""
    print(f"Combining {len(vector_results)} vector + {len(keyword_results)} keyword results")
    
    keyword_map = {}
    for result in keyword_results:
        content = result["content"]
        keyword_map[content] = result["score"]
    
    merged = []
    seen_content = set()
    
    for vec in vector_results:
        content = vec["content"]
        if content in seen_content:
            continue
        seen_content.add(content)
        
        vector_score = vec["score"]
        keyword_score = keyword_map.get(content, 0)
        hybrid_score = (alpha * vector_score) + ((1 - alpha) * keyword_score)
        
        merged.append({
            "content": content,
            "file_name": vec["file_name"],
            "vector_score": vector_score,
            "keyword_score": keyword_score,
            "hybrid_score": hybrid_score,
            "search_type": "hybrid"
        })
    
    for kw in keyword_results:
        content = kw["content"]
        if content not in seen_content:
            seen_content.add(content)
            merged.append({
                "content": content,
                "file_name": kw["file_name"],
                "vector_score": 0,
                "keyword_score": kw["score"],
                "hybrid_score": (1 - alpha) * kw["score"],
                "search_type": "hybrid"
            })
    
    merged_sorted = sorted(merged, key=lambda x: x["hybrid_score"], reverse=True)
    print(f"Hybrid search returning top {min(limit, len(merged_sorted))} results")
    return merged_sorted[:limit]