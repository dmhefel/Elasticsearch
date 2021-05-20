from typing import List
import argparse

from metrics import Score
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Match, MatchAll, ScriptScore, Ids, Query
from elasticsearch_dsl.connections import connections
from embedding_service.client import EmbeddingClient
from utils import parse_wapo_topics
from numpy import dot
from numpy.linalg import norm
connections.create_connection(hosts=["localhost"], timeout=100, alias="default")

def generate_script_score_query(query_vector: List[float], vector_name: str) -> Query:
    """
    generate an ES query that match all documents based on the cosine similarity
    :param query_vector: query embedding from the encoder
    :param vector_name: embedding type, should match the field name defined in BaseDoc ("ft_vector" or "sbert_vector")
    :return: an query object
    """
    q_script = ScriptScore(
        query={"match_all": {}},  # use a match-all query
        script={  # script your scoring function
            "source": f"cosineSimilarity(params.query_vector, '{vector_name}') + 1.0",
            # add 1.0 to avoid negative score
            "params": {"query_vector": query_vector},
        },
    )
    return q_script

def matchall():
    return MatchAll()# a query that matches all documents

def matchTitle(query_text):
    #q_basic = Match(
    #    title={"query": "D.C"}
    #)  # a query that matches "D.C" in the title field of the index, using BM25 as default
    return Match(title={"query": query_text})

def matchContent(query_text):
    return Match(
        content={"query": query_text}
    )

def matchID(vals):
    #q_match_ids = Ids(values=[1, 3, 2])  # a query that matches ids
    return Ids(values=vals)

def matchvector(query_text, vector_name):
    encoder = EmbeddingClient(host="localhost", embedding_type=vector_name)
    #query_text = ["students pursue college education"]
    query_vector = encoder.encode([query_text], pooling="mean").tolist()[
        0
    ]  # get the query embedding and convert it to a list
    q_vector = generate_script_score_query(
        query_vector, vector_name
    )  # custom query that scores documents based on cosine similarity
    return q_vector


    #
    # q_c = (
    #     q_match_ids & q_basic
    # )  # you can also have a compound query by using logic operators on multiple queries

def vectorranking(vector_name, query_vector, x):
    ranking = []
    for i in range(len(x)):
        a = query_vector
        b = x[i].to_dict()[vector_name]
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        ranking.append((cos_sim, i))
    ranking.sort(reverse=True)
    #print(ranking)
    results = []
    for (z,y) in ranking:
        #print(y)
        results.append(x[y])
    return results


    #print("RERANKED")


def search(index: str, query: Query, topk: int) -> None:
    s = Search(using="default", index=index).query(query)[
        :topk
    ]  # initialize a query and return top five results
    response = s.execute()
    for hit in response:
        print(
            hit.meta.id, hit.meta.score, hit.title, hit.annotation, hit.content, sep="\t"
        )  # print the document id that is assigned by ES index, score and title
    return response

if __name__ == "__main__":
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_name",
        required=True,
        type=str,
        help="name of the ES index",
    )
    parser.add_argument(
        "--topic_id",
        required=True,
        type=str,
        help="Topic ID to search",
    )
    parser.add_argument(
        "--query_type",
        required=True,
        type=str,
        help="Narration, Title, or Description",
    )
    parser.add_argument(
        "--top_k",
        required=True,
        type=int,
        help="top_k docs to return",
    )
    parser.add_argument(
        "--vector_name",
        required=False,
        type=str,
        help="name of the vector type (sbert_vector or ft_vector)",
    )
    parser.add_argument(
        "-u",
        action="store_true",
        help="if no vectors used",
    )
    args = parser.parse_args()
    #process xml file
    topics = parse_wapo_topics('pa5_data/topics2018.xml')
    #get type
    if args.query_type == 'narration':
        q_type = 2
    elif args.query_type == 'title':
        q_type = 0
    elif args.query_type == 'description':
        q_type = 1
    #get topic_id[q_type]
    search_term = topics[args.topic_id][q_type]
    #perform the bm25 search
    q_basic = matchContent(search_term)
    index_name = args.index_name
    x = search(
        index_name, q_basic, args.top_k
    )
    relevance = []
    if args.u:
        for hit in x:
            if hit.annotation == args.topic_id + "-2":
                relevance.append(2)
            elif hit.annotation == args.topic_id + "-1":
                relevance.append(1)
            else:
                relevance.append(0)
        score = Score.eval(relevance, args.top_k)
        print(score.ndcg)
    else:
        # if using vectors
        if args.vector_name:
            if args.vector_name.startswith('s'):
                vector_name = 'sbert_vector'
                encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
            elif args.vector_name.startswith('f'):
                vector_name = 'ft_vector'
                encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")

        query_vector = encoder.encode([search_term], pooling="mean").tolist()[
            0
        ]  # get the query embedding and convert it to a list
        #
        results = vectorranking(vector_name, query_vector, x)
        id = args.topic_id
        for hit in results:
            # print(hit.meta.id, hit.meta.score, hit.title, hit.annotation, sep = "\t")
            if hit.annotation == id + "-2":
                relevance.append(2)
            elif hit.annotation == id + "-1":
                relevance.append(1)
            else:
                relevance.append(0)
        score = Score.eval(relevance, args.top_k)
        # print(score.ndcg)
        print(score.ndcg)
    # search(
    #      args.index_name, query, args.top_k
    # )  # search, change the query object to see different results






