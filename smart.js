function forwardPass(input, weights, biases) {
  let outputs = [];

  for (let layer of weights) {
    let layerInputs =
      layer === weights[0] ? input : outputs[outputs.length - 1];
    let weightedSums = layerInputs.map((input, i) =>
      Array.from(layer[i]).reduce((sum, weight) => sum + input * weight, 0),
    );
    let activations = weightedSums.map(activationFunction); // Apply activation function
    outputs.push(activations);
  }

  return outputs;
}

function backpropagate(outputs, weights, targets, learningRate) {
  let errors = [];

  // Output layer errors
  errors.push(
    outputs[outputs.length - 1].map((output, i) => target[i] - output),
  );

  // Hidden layer errors (backpropagation)
  for (let i = weights.length - 2; i >= 0; i--) {
    let layerErrors = [];
    for (let j = 0; j < weights[i][0].length; j++) {
      let error = 0;
      for (let k = 0; k < weights[i + 1].length; k++) {
        error += weights[i + 1][k][j] * errors[errors.length - 1][k];
      }
      layerErrors.push(error * activationFunctionDerivative(outputs[i][j]));
    }
    errors.push(layerErrors);
  }

  // Update weights and biases
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < weights[i].length; j++) {
      for (let k = 0; k < weights[i][j].length; k++) {
        let output = i === 0 ? input[k] : outputs[i - 1][k];
        weights[i][j][k] -=
          learningRate * errors[errors.length - 1 - i][j] * output;
      }
      weights[i][j].push(learningRate * errors[errors.length - 1 - i][j]); // Bias update
    }
  }
}

// Example usage (assuming weights, biases, input, and target are defined):
let outputs = forwardPass(input, weights, biases);
backpropagate(outputs, weights, target, learningRate);
