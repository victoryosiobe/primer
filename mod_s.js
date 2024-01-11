const start = (() => {
  const vocabulary = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-']; // Replace with your full vocabulary
  const vocabSize = vocabulary.length;

  function createInputRepresentation(word) {
    const inputRep = Array(vocabSize).fill(1); //instead of 0 to make input to model okay as 0 turns other stuffs to 0
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
    this.MOBJ = { //we could make the random diffrent. As they are, they  are same in their respective arrays.
      0: {v: Array(inputSize).fill(0)},
      1: {w: Array(hidden1Size).fill(this.random()), b: Array(hidden1Size).fill(this.random()), v: Array(hidden1Size).fill(0)},
      2: {w: Array(hidden2Size).fill(this.random()), b: Array(hidden2Size).fill(this.random()), v: Array(hidden2Size).fill(0)},
      3: {w: Array(outputSize).fill(this.random()), b: Array(outputSize).fill(this.random()), v: Array(outputSize).fill(0)}
      }
    }

    predict(word){
       // Generate input representation with positional encoding
      const inputRep = createInputRepresentation(word);
      const MOBJ = this.MOBJ
      const actiFn = this.actiFn
      let output
      let i = 1
      const mIL = Object.values(MOBJ).length

      //put input values in model input node
      for(let j = 0; j < inputRep.length; j++){
        MOBJ[0].v[j] = inputRep[j]
      }
      
      while(i < mIL){
      //l: layer, c: current, p: previous, w: weight, b: bias, v: value
        const cl = MOBJ[i]
        const pl = MOBJ[i - 1]
        const clw = cl.w
        const clb = cl.b
        const clv = cl.v
        const plw = pl.w
        const plb = pl.b
        const plv = pl.v
        let tempv = 0
        const I = i + 1

        for(let k = 0; k < clv.length; k++){
          for(let l = 0; l < plv.length; l++){
            tempv += plv[l] * clw[k] + clb[k]
            // console.log(tempv, "-f")
          }
          I === mIL ? MOBJ[i].v[k] = actiFn(tempv, 'relu') : MOBJ[i].v[k] = actiFn(tempv, 'sigmoid')
        }
        //console.log(MOBJ[i].v);
        I === mIL ? output = MOBJ[i].v : 0
        i++
       }
      return output
    }

    actiFn(x, type){
      if(type === 'relu') return Math.max(0, x)
      if(type === 'sigmoid') return 1/(1 + Math.exp(-x))
    }

    random(min = 0.0000001, max = 0.0003){
       //Returns a random number between min (inclusive) and max (exclusive)
         return Math.random() * (max - min) + min;
    }
  }

  // Example usage
  const model = new SubwordModel(vocabSize, 200, 100, 3);
  const word = "attend";
  const splitIndex = model.predict(word);
  return `Predicted split index for '${word}': ${splitIndex}`
})
module.exports = start
