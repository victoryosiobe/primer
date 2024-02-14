//We use this file to pipe random words from english words vocabulary.
const tp = performance.now();
const { spawn } = require("child_process");
const trainDataFormat = require("./dataset/trainDataFormat.js");
const fs = require("fs");

let dictionary = "";
let batchSet = new Set();
let dataBatch = 100;
const trainSetupFile = "./dataset/trainSetup";

const trainSetupWords = trainDataFormat(
  fs.readFileSync(trainSetupFile, "utf8"),
).map((_) => _[0]); //collect first value—word value, not target index together.

const pythonProcess = spawn("python", [
  "node_modules/en-words/read_english_dictionary.py",
]);

pythonProcess.stdout.on("data", (data) => {
  dictionary += data;
});

pythonProcess.stderr.on("data", (data) => {
  console.error(`Error from Python: ${data}`);
});

pythonProcess.on("close", (code) => {
  //console.log(`Python process exited with code ${code}`);
  dictionary = dictionary.split(" ");

  let i = 1;
  while (i <= dataBatch) {
    const k = Math.floor(Math.random() * dictionary.length);
    select = dictionary[k];
    if (trainSetupWords.includes(select)) {
      i--;
      continue; //run same iteration immediately from here
    }
    batchSet.add(select);
    if (i + 1 > dataBatch && batchSet.size !== dataBatch) {
      i--; //to try to run same iteration again and again until required batch size of unique element is met.
    }
    i++;
  }
  batchSet = [...batchSet].join("\n"); //sepetate with newline, so i can add target indexes whenever.

  fs.appendFileSync(
    trainSetupFile,
    "\n--New--Untarget--Data--\n" + batchSet,
    "utf8",
  );

  console.log("performace", performance.now() - tp);
});
