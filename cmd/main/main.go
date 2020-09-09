package main

import (
	"encoding/csv"
	"fmt"
	"github.com/rubenwo/NeuralNetworkGo/internal/nn"
	"log"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

)

type data struct {
	inputs  []float64
	outputs []float64
}

var trainingData []data

func main() {
	rand.Seed(time.Now().Unix())

	f, err := os.Open("./assets/trainingdata.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	for _, record := range records {
		in0, err := strconv.Atoi(record[0])
		if err != nil {
			log.Fatal(err)
		}
		in1, err := strconv.Atoi(record[1])
		if err != nil {
			log.Fatal(err)
		}
		out, err := strconv.Atoi(record[2])
		if err != nil {
			log.Fatal(err)
		}
		trainingData = append(trainingData, data{inputs: []float64{float64(in0) / 4000, float64(in1) / 4}, outputs: []float64{float64(out) / 600000}})
	}

	brain := nn.NewBrain(2, 4, 1)
	var wg sync.WaitGroup
	//rand.Seed(time.Now().Unix())
	//Training
	fmt.Println("Started training on:", runtime.NumCPU(), "cpus")
	for z := 0; z < 1; z++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for i := 0; i < 1_000_000; i++ {
				d := trainingData[rand.Intn(len(trainingData))]
				brain.Train(d.inputs, d.outputs)
			}
		}()

	}
	wg.Wait()

	resArr, err := brain.Predict([]float64{1650.0 / 4000, 3 / 4})
	res := resArr[1] * 600000
	fmt.Println("Expecting:", 289221, "Got:", res)

	//for _, val := range trainingData {
	//	res, err := brain.Predict(val.inputs)
	//	if err != nil {
	//		log.Fatal(err)
	//	}
	//	fmt.Println("Expecting:", val.outputs, "Got:", res[1])
	//}
}
