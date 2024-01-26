const fs = require('fs')
const path = require('path')

const dest = 'params.json'

function serialize(data, file = path.join(__dirname, dest)) {
  if (fs.existsSync(file)) {
    saveFn(file, data)
    console.log(`...Serialized to ${file}...`)
  }
  else {
   fs.writeFileSync(file, JSON.stringify([], null, '\t'), 'utf-8')
    saveFn(file, data)
    console.log('File created and data serialized successfully.')
  }
}

function saveFn(file, data) {
  const tempArr = []
  tempArr.push(data)
    let fileContent = fs.readFileSync(file, 'utf8')
  fileContent = JSON.parse(fileContent)
  fileContent.push(tempArr)
  fileContent = fileContent.flat()
  fs.writeFileSync(file, JSON.stringify(fileContent, null, '\t'), 'utf8')
}

function deserialize(file = path.join(__dirname, dest)){
 return JSON.parse(fs.readFileSync(file, 'utf8')).pop() 
}
module.exports = {serialize, deserialize}
