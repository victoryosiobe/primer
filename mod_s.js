const fs = require("fs");
const trainDataFormat = require("./dataset/trainDataFormat.js");
const { serialize, deserialize } = require("./serializer.js");
const start = () => {
  const vocabulary = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "-",
  ]; // Replace with your full vocabulary
  const vocabSize = vocabulary.length;

  function createInputRepresentation(word) {
    const inputRep = Array(vocabSize).fill(0);
    for (let i = 0; i < word.length; i++) {
      const charIndex = vocabulary.indexOf(word[i]);
      const positionalValue = i + 1; // Start positional encoding from 1
      inputRep[charIndex] = charIndex + positionalValue;
    }
    return inputRep;
  }

  class SubwordModel {
    constructor(inputSize, hidden1Size, hidden2Size, outputSize) {
      // Initialize weights and biases randomly
      this.MOBJ =
        deserialize(); /*{ //we could make the random diffrent. As they are, they  are same in their respective arrays.
      0: {v: Array(inputSize).fill(0)},
      1: {w: Array(hidden1Size).fill(0).fill(0).map(_ => this.randomParams()), b: Array(hidden1Size).fill(0).map(_ => this.randomParams()), v: Array(hidden1Size).fill(0)},
      2: {w: Array(hidden2Size).fill(0).map(_ => this.randomParams()), b: Array(hidden2Size).fill(0).map(_ => this.randomParams()), v: Array(hidden2Size).fill(0)},
      3: {w: Array(outputSize).fill(0).map(_ => this.randomParams()), b: Array(outputSize).fill(0).map(_ => this.randomParams()), v: Array(outputSize).fill(0)}
      }*/
    }

    train() {
      //Loop:
      //FeedForward First: Feed In Sample From dataset, get output.
      //Do Stupid Maths, Calculate Loss, Cost by how Output Deviated from sample target.
      //Do uome Backpropagation. Serialize Model.
      //Display Process to Frontend, Per Loop, or Epochs.
      //Implement Logics, to continue looping, or stopping.
      //Implemnt Logic to transfer training samples from dataset file, to usedatset file.
      //Then we Implement finetuning.
    }

    predict(word) {
      // Generate input representation with positional encoding
      const inputRep = createInputRepresentation(word);
      const MOBJ = this.MOBJ;
      const actiFn = this.actiFn;
      let output;
      let i = 1;
      const mIL = Object.values(MOBJ).length;

      //put input values in model input node
      for (let j = 0; j < inputRep.length; j++) {
        MOBJ[0].v[j] = inputRep[j];
      }

      while (i < mIL) {
        //l: layer, c: current, p: previous, w: weight, b: bias, v: value
        const cl = MOBJ[i];
        const pl = MOBJ[i - 1];
        const clw = cl.w;
        const clb = cl.b;
        const clv = cl.v;
        //const plw = pl.w
        //const plb = pl.b
        const plv = pl.v;
        let tempv = 0;
        const I = i + 1;

        for (let k = 0; k < clv.length; k++) {
          for (let l = 0; l < plv.length; l++) {
            tempv += plv[l] * clw[k] + clb[k];
            // console.log(tempv, "-f")
          }
          // I === mIL ?
          MOBJ[i].v[k] = actiFn(tempv, "relu");
          //: MOBJ[i].v[k] = actiFn(tempv, 'sigmoid')
        }
        //console.log(MOBJ[i].v);
        I === mIL ? (output = this.softmax(MOBJ[i].v)) : 0;
        i++;
      }
      return output;
    }

    softmax(x) {
      return Math.round(x);
    }

    actiFn(x, type) {
      if (type === "relu") return Math.max(0, x);
      if (type === "sigmoid") return 1 / (1 + Math.exp(-x));
    }

    randomParams(min = 0.000001, max = 0.0015) {
      //Returns a random number between min (inclusive) and max (exclusive)
      return Math.randomParams() * (max - min) + min;
    }

    dataSet(path = "./dataset/trainSetup") {
      function readFileContent(filePath) {
        try {
          const data = fs.readFileSync(filePath, "utf8");
          return data;
        } catch (error) {
          console.error("Error reading the file:", error);
          return null;
        }
      }

      // Example usage

      const fileContent = readFileContent(path);

      if (fileContent !== null) {
        return trainDataFormat(fileContent.toString());
      }
    }

    learnRate = 0.01;
    batchSize = 3;
    patience = 5; //how much iteration to continue when model is not permorming well
    sampleSize = this.dataSet().length;
  }

  // Example usage
  const model = new SubwordModel(vocabSize, 215, 107, 1);
  // serialize(model.MOBJ)
  const word = "extend";
  console.log("Rep: ", createInputRepresentation(word));
  const splitIndex = model.predict(word);
  // // console.log(model.sampleSize, "xup");
  // console.log(model.dataSet());
  return `Predicted split index for '${word}': ${splitIndex}`;
};
module.exports = start;
