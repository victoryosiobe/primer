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
    const inputRep = Array(vocabSize).fill(1);
    let tempState = [];
    for (let i = 0; i < word.length; i++) {
      if (tempState.includes(word[i])) continue;
      const positionalValue = vocabulary.indexOf(word[i]);
      const freq = word.split("").filter((_) => _ === word[i]).length;
      const smoother = Math.cos(positionalValue + 1);
      inputRep[positionalValue] = smoother + freq + (i + 1); //+1 incase if it's zero
      tempState.push(word[i]); //store so we don't encounter it again
    }
    for (const v in inputRep) {
      inputRep[v] += inputRep.reduce((pv, cv) => pv + cv) / inputRep.length; //to create diverse-re-reproduced values, the mean is calculated after updating just a value; thus this changes mean, and when re-calculated it gives a different value to next array-value in loop. So all values are unique, but constant. What a pattern to learn!
    }

    return inputRep;
  }

  class SubwordModel {
    constructor(inputSize, hidden1Size, hidden2Size, outputSize) {
      // Initialize weights and biases randomly
      this.MOBJ = deserialize();
      //this.randomInit();
    }

    train(dataset = this.dataSet(), epochs = 10000) {
      //Loop:
      //FeedForward First: Feed In Sample From dataset, get output.
      //Do Stupid Maths, Calculate Loss, Cost by how Output Deviated from sample target.
      //Do uome Backpropagation. Serialize Model.
      //Display Process to Frontend, Per Loop, or Epochs.
      //Implement Logics, to continue looping, or stopping.
      //Implemnt Logic to transfer training samples from dataset file, to usedatset file.
      //Then we Implement finetuning.
      let i = 0;
      let j = 0;

      while (i < epochs) {
        if (this.cost <= this.costThres) {
          console.log("Model Trained...");
          //break;
        }
        while (j < dataset.length) {
          const word = dataset[j][0];
          const target = dataset[j][1];
          const outP = this.predict(word, "verbose");
          this.lossCostFn(outP.out, target, i); //sets loss and cost method
          if (this.cost <= this.costThres) break;
          this.updateParam(outP.llov);
          j++;
        }
        //update learningrate
        i++;

        // const word = dataset[i][0];
        // const target = dataset[i][1];
        // let j = 0;
        // let patience = this.patience;
        // while (j < epochs) {
        //   if (patience === 0) break;
        //   this.iterations++;
        //   const output = this.predict(word); //get index value
        //   this.lossCostFn(output, target, this.iterations);
        //   //Here we implement Backpropagation. Then seriakize model
        //   if ("modelDidNotImprove") patience--;
        //   j++;
        // }
        //Here we remove the word from dataset and accumulate them in usedData fike
        //Here we save this.iterations, learning rate, cost, loss, to metaData file and reload whenever needed, done like this to backup when i end session, or node process.
      }
    }
    updateParam(llov) {
      const model = this.MOBJ;
      const copyllov = llov.slice();

      const gradient = [this.loss * activaDeriveRELU(llov.pop())];
      const j = Object.values(this.MOBJ).length - 1;
      let k = 0;
      for (const i of llov.reverse()) {
        const hiddenLayerError = model[j - 1].w.map(
          (weight) => gradient[gradient.length - 1] * weight,
        );
        const hiddenLayerGradient = hiddenLayerError
          .map((err) =>
            i.map((v) => activaDeriveRELU(v)).map((acti) => err * acti),
          )
          .reduce((acc, v) => acc * v, 1);
        gradient.push(hiddenLayerGradient);
      }
      gradient.reverse();
      while (k < j) {
        this.MOBJ[k + 1].w = this.MOBJ[k + 1].w.map(
          (weight, l) => weight * this.learnRate * gradient[k] * copyllov[k][l],
        );
        k++;
      }
      console.log(this.MOBJ);
      function activaDeriveRELU(x) {
        return x > 0 ? 1 : 0;
      }
      //function lossDerivateMSE(x) {}
    }

    predict(word, mode) {
      const llov = []; //layer layer output values, for storing all layer outputs. used in trsining
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
        llov.push(MOBJ[i].v);
        I === mIL ? (output = this.softmax(MOBJ[i].v)) : 0;
        i++;
      }
      //console.log("------------", llov);
      return mode === "verbose" ? { out: output, llov: llov } : output;
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
      return Math.random() * (max - min) + min;
    }

    randomInit() {
      //we could make the random diffrent. As they are, they  are same in their respective arrays.
      return {
        0: { v: Array(inputSize).fill(0) },
        1: {
          w: Array(hidden1Size)
            .fill(0)
            .fill(0)
            .map((_) => this.randomParams()),
          b: Array(hidden1Size)
            .fill(0)
            .map((_) => this.randomParams()),
          v: Array(hidden1Size).fill(0),
        },
        2: {
          w: Array(hidden2Size)
            .fill(0)
            .map((_) => this.randomParams()),
          b: Array(hidden2Size)
            .fill(0)
            .map((_) => this.randomParams()),
          v: Array(hidden2Size).fill(0),
        },
        3: {
          w: Array(outputSize)
            .fill(0)
            .map((_) => this.randomParams()),
          b: Array(outputSize)
            .fill(0)
            .map((_) => this.randomParams()),
          v: Array(outputSize).fill(0),
        },
      };
    }

    dataSet(path = "./dataset/trainSetup") {
      function readFileContent(filePath) {
        try {
          const data = fs.readFileSync(filePath, "utf8");
          return data;
        } catch (error) {
          console.error("Error reading the file:", error);
          throw new Error("FAILED TO LOAD DATASET");
        }
      }

      // Example usage

      const fileContent = readFileContent(path);

      if (fileContent !== null) {
        return trainDataFormat(fileContent.toString());
      }
    }
    lossCostFn(output, target, samples_done) {
      // Loss function - Mean Squared Error (MSE)
      const v = Math.pow(target - output, 2);
      this.losses += v;
      this.cost = this.losses / samples_done; // Cost function - Mean Squared Error (MSE) over entire training duration.
      this.loss = v; //deviation from target, calculated on every epoch.
    }
    loss = 0;
    losses = 0;
    cost = 0;
    costThres = 0.07;
    iterations = 0;
    learnRate = 0.01;
    //batchSize = 3;
    patience = 5; //how much iteration to continue when model is not permorming well
    sampleSize = this.dataSet().length;
  }

  // Example usage
  const model = new SubwordModel(vocabSize, 215, 107, 1);
  model.train();
  // serialize(model.MOBJ)
  const word = "extend";
  //console.log("Rep: ", createInputRepresentation(word));
  const splitIndex = model.predict(word);
  // // console.log(model.sampleSize, "xup");
  // console.log(model.dataSet());
  return `Predicted split index for '${word}': ${splitIndex}`;
};
module.exports = start;
