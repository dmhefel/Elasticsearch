from flask import Flask, render_template, request
from evaluate import matchContent, vectorranking, search, matchID
app = Flask(__name__)
query = ""
k = 20 #top_k can be changed
results = []
index_name = ""
# home page
@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["POST","GET"])
def results():
    global query
    global results
    global index_name
    query_text = request.form["query"]  # Get the raw user query from home page

    if query_text:
        query = query_text
        option = request.form['options']  # type of button

        #perform search
        q_basic = matchContent(query)
        #if using default
        index_name = "wapo_docs_50k"
        #custom
        if option == "bmcus":
            index_name = "custom_idx"
        #get results of search
        results = search(index_name, q_basic, k)
        results = [{i.meta.id: i.to_dict()} for i in results]

        #if reranking with vector
        if option.endswith("3") or option.endswith("4"):
            if option == "sbert":
                vector_name = 'sbert_vector'
                encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
            elif option == "ft":
                vector_name = 'ft_vector'
                encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")
            query_vector = encoder.encode([query], pooling="mean").tolist()[
                           0
                       ]
            results = vectorranking(vector_name, query_vector, x)
        else:
            if query_text == "":
                query_text = "EMPTY"
        results = [results[i:i + 8] for i in range(0, len(results), 8)]
        # first page
        page = 1
        # starting number for listing
        s = ((page - 1) * 8 + 1)
        # next page num
        next = page + 1
        # if there is a next page
        hasnext = False
        if len(results) > page:
            hasnext = True
        res = []
        if results:
            res = results[page - 1]


        return render_template("results.html", query=query_text, result=res, page=page, next_page=next, start=s, hasnext=hasnext)  # add variables as you wish

# "next page" to show more results
@app.route("/results/<query_text>/<int:page_id>", methods=["POST", "GET"])
def next_page(page_id, query_text):
    global query
    global results
    page = page_id
    # starting number for listing
    s = ((page - 1) * 8 + 1)
    # next page num
    next = page + 1
    # sets variable for if there is next page
    hasnext = False
    if len(results) > page:
        hasnext = True
    if results:
        res = results[page - 1]
    return render_template("results.html", query=query_text, result=res, page=page, next_page=next,
                           start=s,
                           hasnext=hasnext)  # add variables as you wish


# document page
@app.route("/doc_data/<int:doc_id>")
def doc_data(doc_id):
    q_match_ids = matchID([doc_id])
    x = search(
        index_name, q_match_ids, 1
    )
    x = x[0]
    auth = x.author
    date = x.date
    title = x.title
    txt = x.content


    return render_template("doc.html", Text=txt, Author=auth, Title=title, Date=date)  # add variables as you wish


if __name__ == "__main__":
    app.run(debug=True, port=5000)


