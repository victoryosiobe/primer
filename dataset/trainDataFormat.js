module.exports = (v = "") => {
  const initArr = v.split("\n").filter((_) => _ !== "");
  if (!initArr) throw Error("training data is most likely empty.");
  let arrOK = [];
  for (let values of initArr) {
    if (values.startsWith("--")) break;
    arrOK.push(values);
  }

  arrOK = arrOK.map((_) => _.split(/\s/));

  return arrOK;
};
