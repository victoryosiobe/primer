const start = (() => {
  const vocabulary = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-']; // Replace with your full vocabulary
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
    constructor(vocabSize, hidden1Size, hidden2Size) {
      // Initialize weights and biases randomly
      this.w1 = Array(vocabSize).fill(0).map(() => Array(hidden1Size).fill(Math.random()));
      this.b1 = Array(hidden1Size).fill(0);
      this.w2 = Array(hidden1Size).fill(0).map(() => Array(hidden2Size).fill(Math.random()));
      this.b2 = Array(hidden2Size).fill(0);
      this.w3 = Array(hidden2Size).fill(0).map(() => Array(1).fill(Math.random()));
      this.b3 = Array(1).fill(0);
    }

    predict(word) {
      // Generate input representation with positional encoding
      const inputRep = createInputRepresentation(word);

      // Feedforward through hidden layers with ReLU activation
      const h1 = inputRep.map((x, i) => this.actiFn(x * this.w1[i].reduce((sum, w) => sum + w) + this.b1[i], 'sigmoid'));
      const h2 = h1.map((x, i) => this.actiFn(x * this.w2[i].reduce((sum, w) => sum + w) + this.b2[i], 'sigmoid'));
      const output = h2.map((x, i) => this.actiFn(x * this.w3[i][0] + this.b3[0], 'relu'));

      return output[0]; // Return predicted split index
    }

    actiFn(x, type){
     if(type === 'relu') return Math.max(0, x)
     if(type === 'sigmoid') return 1/(1 + Math.exp(-x))
    }
  }

  // Example usage
  const model = new SubwordModel(vocabSize, 200, 100);
  const word = "attend";
  const splitIndex = model.predict(word);
  return `Predicted split index for '${word}': ${splitIndex}`
})
module.exports = start
