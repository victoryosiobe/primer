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
      0: {w: Array(inputSize).fill(this.random()), b: Array(inputSize).fill(this.random())},
      1: {w: Array(hidden1Size).fill(this.random()), b: Array(hidden1Size).fill(this.random()), v: 0},
      2: {w: Array(hidden2Size).fill(this.random()), b: Array(hidden2Size).fill(this.random()), v: 0},
      3: {w: Array(outputSize).fill(this.random()), b: Array(outputSize).fill(this.random()), v: 0}
      }
    }
    predict(word) {
      // Generate input representation with positional encoding
      const inputRep = createInputRepresentation(word);
      const MOBJ = this.MOBJ
      const actiFn = this.actiFn
      let output
      let i = 0
      const mIL = Object.values(MOBJ).length
      while(i < mIL){
        if(i === 0){
          MOBJ[i + 1].v = inputRep.map((v, j) => actiFn(v * MOBJ[i].w[j] + MOBJ[i].b[j], 'sigmoid'))
          i++
          continue
        }
        // console.log(MOBJ[i].v, i)
        /*  if(i === (mIL - 1)){ // output node operation
            MOBJ[i + 1].v = MOBJ[i].w.map((w, j) => { 
          return actiFn(MOBJ[i].v.map(v => v * w + MOBJ[i].b[j]).reduce((acc, v) => acc + v, 0), 'relu')
          })
            break
        }
      */
        if(!(MOBJ[i + 1] === undefined)){
          MOBJ[i + 1].v = MOBJ[i].w.map((w, j) => {
       return  actiFn(MOBJ[i].v.map(v => v * w + MOBJ[i].b[j]).reduce((acc, v) => acc + v, 0), 'sigmoid')
       })
        } else break
        i++
      }
      return Object.values(MOBJ).pop().v; // Return predicted split index
      
    }

    actiFn(x, type){
     if(type === 'relu') return Math.max(0, x)
     if(type === 'sigmoid') return 1/(1 + Math.exp(-x))
    }

    random(){
      return Math.random()
    }
  }

  // Example usage
  const model = new SubwordModel(vocabSize, 200, 100, 3);
  const word = "attend";
  const splitIndex = model.predict(word);
  return `Predicted split index for '${word}': ${splitIndex.length}`
})
module.exports = start
