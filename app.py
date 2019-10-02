from flask import Flask, jsonify
import os


from inception1 import maybe_download, Inception
data_dir = "inception/"
app = Flask(__name__)
maybe_download()
model = Inception()


@app.route('/')
def hello():
    image_path = os.path.join(data_dir, 'arshiya.jpeg')

    # Use the Inception model to classify the image.
    pred = model.classify(image_path=image_path)

    return jsonify(model.print_scores(pred=pred, k=10))

if __name__ == '__main__':
    app.run()