const { flowerData } = require("./data");
const tf = require("@tensorflow/tfjs-node");

const labelList = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
const labelTensor = [];
const dataTensor = [];

const test = [6.3, 2.8, 5.1, 1.5];
const processData = () => {
  flowerData.forEach(flower => {
    const arr = [];
    flower.forEach(item => {
      if (typeof item === "string") {
        labelTensor.push(labelList.indexOf(item));
        return;
      }
      arr.push(item);
    });
    dataTensor.push(arr);
  });
};

const init = () => {
  processData();
  const model = tf.sequential();
  const hiddenLayer = tf.layers.dense({
    units: 3,
    inputShape: [4],
    activation: "sigmoid"
  });
  model.add(hiddenLayer);

  const outputLayer = tf.layers.dense({
    units: 3,
    activation: "softmax"
  });

  model.add(outputLayer);

  const optimizer = tf.train.sgd(0.2);

  model.compile({
    optimizer,
    loss: "categoricalCrossentropy"
  });

  const tensorLabel = tf.tensor1d(labelTensor, "int32");
  const xs = tf.tensor2d(dataTensor);
  const ys = tf.oneHot(tensorLabel, 3);

  console.log(xs.shape, ys.shape);

  const train = async () => {
    try {
      let i = 0;
      const config = {
        shuffle: true,
        epochs: 1
      };
      for (i = 0; i < 10000; i++) {
        await model.fit(xs, ys, config);
      }
    } catch (e) {
      console.log("error: ", e);
    }
  };

  train().then(() => {
    const trainTensor = tf.tensor2d([test]);
    console.log(trainTensor.shape);
    model.predict(trainTensor).print();
  });
};

init();
