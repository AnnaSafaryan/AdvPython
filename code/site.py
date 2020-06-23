"""
Постенькая страничка для поиска ближайших в нашем корпусе (included=1)
"""

from flask import Flask, request

from monocorp_search import search

app = Flask(__name__)

template = open('templates/main_template.html', encoding='utf-8').read()
table_template = open('templates/table_template.html', encoding='utf-8').read()
tableraw_template = open('templates/tableraw_template.html', encoding='utf-8').read()


def form_table(rating):
    tableraws = ''
    for i, res in enumerate(rating):
        # print(type(res))
        raw = tableraw_template.format(i+1, res['title'], res['lang'], res['sim'])
        tableraws += raw
    table = table_template.format(tableraws)
    return table

lang = 'cross'
mapping_path = '../texts_wiki/mapping.json'
vectors_path_dict = {
    'vecmap': '../texts_wiki/common_vecmap.pkl',
    'muse': '../texts_wiki/common_muse.pkl',
    'proj': './texts_wiki/common_trans.pkl',
    'trans': '../texts_wiki/common_proj.pkl'
    }


@app.route('/')
def hw():
    return template %'Hello!'


@app.route('/find', methods=['GET'])
def find():
    print(request.args)
    n = int(request.args.get('top'))
    # print(n, type(n))
    corpus_vectors_path = vectors_path_dict[request.args.get('method')]
    rating, verbosed_rating, _ = search(request.args.get('title'), lang, mapping_path,
                                        corpus_vectors_path, top=n, verbose=0)
    # print(verbosed_rating)
    # print(type(rating), rating)
    table = form_table(rating)
    return template %table


if __name__ == "__main__":
    app.run(port=8081, debug=True)

