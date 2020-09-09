package nn

import (
	"encoding/json"
	"github.com/rubenwo/NeuralNetworkGo/internal/matrix"
	"log"
	"math"
	"sync"

)

//ActivationFunction ...
type ActivationFunction interface {
	getFunc() func(float64, int, int) float64
	getFuncD() func(float64, int, int) float64
}

//Sigmoid ...
type Sigmoid struct{}

func (s *Sigmoid) getFunc() func(float64, int, int) float64 {
	return func(x float64, _, _ int) float64 { return 1 / (1 + math.Exp(-x)) }
}
func (s *Sigmoid) getFuncD() func(float64, int, int) float64 {
	return func(y float64, _, _ int) float64 { return y * (1 - y) }
}

//TanH ...
type TanH struct{}

func (t *TanH) getFunc() func(float64, int, int) float64 {
	return func(x float64, _, _ int) float64 { return math.Tanh(x) }
}

func (t *TanH) getFuncD() func(float64, int, int) float64 {
	return func(y float64, _, _ int) float64 { return 1 - (y * y) }
}

//Brain ...
type Brain struct {
	lock                 sync.Mutex
	in, hidden, out      int
	weigthsIH, weightsHO *matrix.Matrix
	biasH, biasO         *matrix.Matrix
	learningRate         float64
	activationFunction   ActivationFunction
}

//NewBrain ...
func NewBrain(in, hidden, out int) *Brain {
	brain := &Brain{
		in:        in,
		hidden:    hidden,
		out:       out,
		weigthsIH: matrix.New(hidden, in).Randomize(),
		weightsHO: matrix.New(out, hidden).Randomize(),
		biasH:     matrix.New(hidden, 1).Randomize(),
		biasO:     matrix.New(out, 1).Randomize()}

	brain.SetLearningRate(0.1)
	brain.SetActivationFunction(&Sigmoid{})

	return brain
}

//Copy ...
func Copy(b *Brain) *Brain {
	brain := &Brain{
		in:        b.in,
		hidden:    b.hidden,
		out:       b.out,
		weigthsIH: b.weigthsIH.Copy(),
		weightsHO: b.weightsHO.Copy(),
		biasH:     b.biasH.Copy(),
		biasO:     b.biasO.Copy()}

	brain.SetLearningRate(b.learningRate)
	brain.SetActivationFunction(b.activationFunction)

	return brain
}

//Deserialize ...
func Deserialize(data []byte) (*Brain, error) {
	var b Brain
	err := json.Unmarshal(data, &b)
	if err != nil {
		return nil, err
	}
	return &b, nil
}

//SetLearningRate ...
func (b *Brain) SetLearningRate(lr float64) {
	b.learningRate = lr
}

//SetActivationFunction ...
func (b *Brain) SetActivationFunction(a ActivationFunction) {
	b.activationFunction = a
}

//Predict ...
func (b *Brain) Predict(input []float64) ([]float64, error) {
	inputs := matrix.FromArray(input)
	//	fmt.Println("INPUTS:", inputs)
	hidden, err := matrix.DotMultiply(b.weigthsIH, inputs)
	//	fmt.Println("HIDDEN:", hidden)

	if err != nil {
		return nil, err
	}
	hidden, err = hidden.AddMatrix(b.biasH)
	if err != nil {
		return nil, err
	}
	hidden.Map(b.activationFunction.getFunc())
	output, err := matrix.DotMultiply(b.weightsHO, hidden)
	//	fmt.Println("OUTPUTS:", output)

	if err != nil {
		return nil, err
	}
	output.AddMatrix(b.biasO)
	output.Map(b.activationFunction.getFunc())
	return output.ToArray(), nil
}

//Train ...
func (b *Brain) Train(input, target []float64) {
	inputs := matrix.FromArray(input)
	hidden, err := matrix.DotMultiply(b.weigthsIH, inputs)
	if err != nil {
		log.Fatal("1", err)
	}
	hidden, err = hidden.AddMatrix(b.biasH)
	if err != nil {
		log.Fatal("2", err)
	}
	hidden.Map(b.activationFunction.getFunc())

	outputs, err := matrix.DotMultiply(b.weightsHO, hidden)
	if err != nil {
		log.Fatal("3", err)
	}
	outputs, err = outputs.AddMatrix(b.biasO)
	if err != nil {
		log.Fatal("4", err)
	}
	outputs.Map(b.activationFunction.getFunc())

	targets := matrix.FromArray(target)

	outputErrors, err := matrix.Subtract(targets, outputs)
	if err != nil {
		log.Fatal("5", err)
	}

	gradients := matrix.MapStatic(outputs, b.activationFunction.getFuncD())
	gradients, err = gradients.MultiplyMatrix(outputErrors)
	if err != nil {
		log.Fatal("6", err)
	}
	gradients.MultiplyScalar(b.learningRate)

	hiddenT := matrix.Transpose(hidden)
	weightHODeltas, err := matrix.DotMultiply(gradients, hiddenT)
	if err != nil {
		log.Fatal("7", err)
	}

	b.weightsHO, err = b.weightsHO.AddMatrix(weightHODeltas)
	if err != nil {
		log.Fatal("8", err)
	}
	b.biasO, err = b.biasO.AddMatrix(gradients)
	if err != nil {
		log.Fatal("9", err)
	}

	whoT := matrix.Transpose(b.weightsHO)
	hiddenErrors, err := matrix.DotMultiply(whoT, outputErrors)
	if err != nil {
		log.Fatal("10", err)
	}

	hiddenGradient := matrix.MapStatic(hidden, b.activationFunction.getFuncD())
	hiddenGradient, err = hiddenGradient.MultiplyMatrix(hiddenErrors)
	if err != nil {
		log.Fatal("11", err)
	}
	hiddenGradient = hiddenGradient.MultiplyScalar(b.learningRate)

	inputsT := matrix.Transpose(inputs)
	weightIHDeltas, err := matrix.DotMultiply(hiddenGradient, inputsT)
	if err != nil {
		log.Fatal("12", err)
	}
	b.weigthsIH, err = b.weigthsIH.AddMatrix(weightIHDeltas)
	if err != nil {
		log.Fatal("13", err)
	}
	b.biasH, err = b.biasH.AddMatrix(hiddenGradient)
	if err != nil {
		log.Fatal("14", err)
	}
}

//Mutate ...
func (b *Brain) Mutate(fn matrix.MappingFunction) {
	b.weigthsIH.Map(fn)
	b.weightsHO.Map(fn)
	b.biasH.Map(fn)
	b.biasO.Map(fn)
}

//Serialize ...
func (b *Brain) Serialize() ([]byte, error) {
	return json.Marshal(b)
}
