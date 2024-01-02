from flask import Flask, render_template, request
from main import get_player

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
async def index():  # Add 'async' keyword here
    if request.method == 'POST':
        print(request.form)
        search_query = request.form['searchQuery']
        print(search_query)
        player = await get_player(search_query)  # Await the get_player coroutine
        print(player)
        return render_template('index.html', searchResults=player)
    return render_template('index.html')

def hello_world():
    return "<p>Hello, World!</p>"