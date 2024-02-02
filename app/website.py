from flask import Blueprint,render_template,abort,url_for,redirect
from jinja2 import TemplateNotFound
import markdown

"""website_home_blueprint = Blueprint('website_home_blueprint', __name__)

@website_home_blueprint.route('/')
def home():
    return render_template("index.html")"""

website_pages_blueprint = Blueprint('website_pages_blueprint', __name__)

@website_pages_blueprint.route('/pages/')
@website_pages_blueprint.route('/pages/<page>')
def show(page):
    try:
        return render_template('%s.html' % page)
    except TemplateNotFound:
        abort(404)

home_blueprint = Blueprint('home_blueprint', __name__)
@home_blueprint.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html><head>
    <title>index</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
    <link rel="stylesheet" href="static/css/katex.min.css">
    <link href="static/css/style.css" rel="stylesheet" type="text/css">
    <script src="https://kit.fontawesome.com/849ddf9236.js" crossorigin="anonymous"></script>
    <script src="static/js/jquery-3.4.1.min.js"></script>
    </head>

    <body for="html-export">
    <div class="mume markdown-preview" id="main">
    %s
    </div>
    <script src="static/js/label/main.js"></script>
    </body></html>
  
    '''
    
    readme_file = open("README.md", "r",encoding="utf-8")
    md_template_string = markdown.markdown(
        readme_file.read(), extensions=[
    'markdown.extensions.extra', 'markdown.extensions.codehilite','markdown.extensions.tables','markdown.extensions.toc']
    )
    return html% md_template_string
