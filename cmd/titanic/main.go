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
	"time"
)

type Data struct {
	PassengerId int
	Survived    int
	Pclass      int
	Name        string
	Sex         string
	Age         float64
	SibSp       int
	Parch       int
	Ticket      string
	Fare        float64
	Cabin       string
	Embarked    string
}

type training struct {
	in  []float64
	out []float64
}

func main() {
	d := readCsv("./assets/train.csv", true)

	rand.Seed(time.Now().UnixNano())
	//rand.Shuffle(len(d), func(i, j int) { d[i], d[j] = d[j], d[i] })

	nFeatures := 8
	epochs := 2_000
	brain := nn.NewBrain(nFeatures, 8, 1)
	brain.SetLearningRate(0.05)

	var trainingData []training
	var valData []training
	split := int(float64(len(d)) * 0.6)

	for _, data := range d[:split] {
		trainingData = append(trainingData, data.convert())
	}
	for _, data := range d[split:] {
		valData = append(valData, data.convert())
	}

	for i := 0; i < epochs; i++ {
		rand.Shuffle(len(trainingData), func(i, j int) { trainingData[i], trainingData[j] = trainingData[j], trainingData[i] })

		for _, t := range trainingData {
			brain.Train(t.in, t.out)
		}
	}

	core := runtime.NumCPU()
	mse := 0.0
	for _, v := range trainingData {
		o, err := brain.Predict(v.in)
		if err != nil {
			log.Fatal(err)
		}
		mse += (v.out[0] - o[1]) * (v.out[0] - o[1])
	}
	fmt.Printf("Core: %d: MSE trainData: %f\n", core, mse/float64(len(trainingData)))

	mse = 0.0
	for _, v := range valData {
		o, err := brain.Predict(v.in)
		if err != nil {
			log.Fatal(err)
		}
		mse += (v.out[0] - o[1]) * (v.out[0] - o[1])
	}
	fmt.Printf("Core: %d: MSE valData: %f\n", core, mse/float64(len(valData)))

	var rec [][]string
	header := []string{"PassengerId", "Survived"}
	rec = append(rec, header)
	testSet := readCsv("./assets/test.csv", false)
	for _, t := range testSet {
		train := t.convert()
		o, err := brain.Predict(train.in)
		if err != nil {
			log.Fatal(err)
		}

		survived := "0"
		if o[1] >= 0.5 {
			survived = "1"
		}

		rec = append(rec, []string{strconv.Itoa(t.PassengerId), survived})
	}

	f, err := os.Create("./assets/generated.csv")
	if err != nil {
		log.Fatal(err)
	}
	writer := csv.NewWriter(f)
	writer.WriteAll(rec)
	writer.Flush()
	f.Close()
}

func calcMeanAge(d []*Data) float64 {
	total := 0.0
	for _, data := range d {
		if data.Age != -1 {
			total += data.Age
		}
	}
	return total / float64(len(d))
}

func normalize(in, low, high float64) float64 {
	return (in - low) / (high - low)
}

func (d *Data) convert() training {
	s := 0
	if d.Sex == "female" {
		s = 1
	}
	pc := []float64{0, 0, 0}
	switch d.Pclass {
	case 1:
		pc[0] = 1
		break
	case 2:
		pc[1] = 1
		break
	case 3:
		pc[2] = 1
		break
	}

	t := training{
		in: []float64{normalize(d.Age, 0.42, 80.0), normalize(d.Fare, 0, 512.3292),
			normalize(float64(d.SibSp), 0, 8), normalize(float64(d.Parch), 0, 6), float64(s)},
		out: []float64{float64(d.Survived)},
	}
	t.in = append(t.in, pc...)
	return t
}

func readCsv(path string, train bool) []*Data {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	if train {
		var d []*Data
		for _, record := range records[1:] {
			id, err := strconv.Atoi(record[0])
			if err != nil {
				fmt.Println(err)
			}
			survived, err := strconv.Atoi(record[1])
			if err != nil {
				fmt.Println(err)
			}
			pclass, err := strconv.Atoi(record[2])
			if err != nil {
				fmt.Println(err)
			}
			age, err := strconv.ParseFloat(record[5], 64)
			if err != nil {
				fmt.Println("Age:", err)
				age = -1
			}
			sib, err := strconv.Atoi(record[6])
			if err != nil {
				fmt.Println(err)
			}
			par, err := strconv.Atoi(record[7])
			if err != nil {
				fmt.Println(err)
			}
			fare, err := strconv.ParseFloat(record[9], 64)
			if err != nil {
				fmt.Println("Fare:", err)
			}
			d = append(d, &Data{
				PassengerId: id,
				Survived:    survived,
				Pclass:      pclass,
				Name:        record[3],
				Sex:         record[4],
				Age:         age,
				SibSp:       sib,
				Parch:       par,
				Ticket:      record[8],
				Fare:        fare,
				Cabin:       record[10],
				Embarked:    record[11],
			})
		}
		meanAge := calcMeanAge(d)
		for _, data := range d {
			if data.Age == -1 {
				data.Age = meanAge
			}
		}
		return d
	}
	var d []*Data
	for _, record := range records[1:] {
		id, err := strconv.Atoi(record[0])
		if err != nil {
			fmt.Println(err)
		}
		pclass, err := strconv.Atoi(record[1])
		if err != nil {
			fmt.Println(err)
		}
		age, err := strconv.ParseFloat(record[4], 64)
		if err != nil {
			fmt.Println("Age:", err)
			age = -1
		}
		sib, err := strconv.Atoi(record[5])
		if err != nil {
			fmt.Println(err)
		}
		par, err := strconv.Atoi(record[6])
		if err != nil {
			fmt.Println(err)
		}
		fare, err := strconv.ParseFloat(record[8], 64)
		if err != nil {
			fmt.Println("Fare:", err)
		}
		d = append(d, &Data{
			PassengerId: id,
			Pclass:      pclass,
			Name:        record[2],
			Sex:         record[3],
			Age:         age,
			SibSp:       sib,
			Parch:       par,
			Ticket:      record[7],
			Fare:        fare,
			Cabin:       record[9],
			Embarked:    record[10],
		})

	}
	meanAge := calcMeanAge(d)
	for _, data := range d {
		if data.Age == -1 {
			data.Age = meanAge
		}
	}
	return d
}
