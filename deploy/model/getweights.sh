wget "https://github.com/sorohere/flickr-dataset/releases/download/v0.2.0/model.zip"
wget "https://github.com/sorohere/flickr-dataset/releases/download/v0.2.0/vocab.zip"

unzip model.zip -d ./model
unzip vocab.zip -d ./model

rm model.zip
rm vocab.zip