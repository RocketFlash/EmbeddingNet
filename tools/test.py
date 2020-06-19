from embedding_net.model import EmbeddingNet
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str,
                        help="path to config file")
    parser.add_argument("--weights", type=str,
                        help="path to trained model weights file")
    parser.add_argument("--encodings", type=str,
                        help="path to trained model encodings file")
    parser.add_argument("--image", type=str, help="path to image file")
    opt = parser.parse_args()

    config_path = opt.config
    weights_path = opt.weights
    encodings_path = opt.encodings
    image_path = opt.image

    model = EmbeddingNet(config_path)
    model.load_model(weights_path)
    model.load_encodings(encodings_path)

    model_prediction = model.predict(image_path)
    print('Model prediction: {}'.format(model_prediction))
