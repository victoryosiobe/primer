function inputHandle (input) {
  function subWordModel(input){
    const model = require("./mod_s.js")
    return model(input)
  }
  //More Functions Here. You can switch model functions

  const sendOut = subWordModel(input)
  return sendOut
}
module.exports = {inputHandle}
