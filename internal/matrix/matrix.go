package matrix

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"time"
)

//MappingFunction ...
type MappingFunction func(val float64, i, j int) float64

//Matrix ...
type Matrix struct {
	rows, cols int
	data       [][]float64
}

//New ...
func New(rows, cols int) *Matrix {
	mat := &Matrix{rows: rows,
		cols: cols,
	}
	a := make([][]float64, rows)
	for i := range a {
		a[i] = make([]float64, cols)
	}
	mat.data = a
	return mat
}

//FromArray ...
func FromArray(arr []float64) *Matrix {
	//	fmt.Println(arr)
	mat := New(len(arr), 1)
	mat.Map(func(val float64, i, j int) float64 {
		return arr[i+j]
	})
	//	fmt.Println(mat)

	return mat
}

//Transpose ...
func Transpose(m *Matrix) *Matrix {
	return New(m.cols, m.rows).Map(func(_ float64, i, j int) float64 {
		return m.data[j][i]
	})
}

//DotMultiply ...
func DotMultiply(a, b *Matrix) (*Matrix, error) {
	if a.cols != b.rows {
		return nil, errors.New("columns of A must match rows of B")
	}

	return New(a.rows, b.cols).Map(func(val float64, i, j int) float64 {
		// Dot product of values in col
		sum := 0.0
		for k := 0; k < a.cols; k++ {
			sum += a.data[i][k] * b.data[k][j]
		}
		return sum
	}), nil
}

//MapStatic ...
func MapStatic(m *Matrix, fn MappingFunction) *Matrix {
	return New(m.rows, m.cols).Map(func(val float64, i, j int) float64 {
		return fn(m.data[i][j], i, j)
	})
}

//Deserialize ...
func Deserialize(data []byte) (*Matrix, error) {
	var mat Matrix
	err := json.Unmarshal(data, &mat)
	if err != nil {
		return nil, err
	}
	return &mat, nil
}

//ToArray ...
func (m *Matrix) ToArray() []float64 {
	arr := make([]float64, m.rows*m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			arr = append(arr, m.data[i][j])
		}
	}
	//	fmt.Println("ARRAY:", arr)
	return arr
}

//Randomize ...
func (m *Matrix) Randomize() *Matrix {
	return m.Map(func(val float64, i, j int) float64 {
		return rand.New(rand.NewSource(time.Now().UnixNano())).Float64()
	})
}

//Copy ...
func (m *Matrix) Copy() *Matrix {
	mat := New(m.rows, m.cols)

	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			mat.data[i][j] = m.data[i][j]
		}
	}
	return mat
}

//Map ...
func (m *Matrix) Map(fn MappingFunction) *Matrix {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			val := m.data[i][j]
			m.data[i][j] = fn(val, i, j)
		}
	}
	return m
}

//MultiplyScalar ...
func (m *Matrix) MultiplyScalar(scalar float64) *Matrix {
	m.Map(func(val float64, i, j int) float64 {
		return val * scalar
	})
	return m
}

//MultiplyMatrix ...
func (m *Matrix) MultiplyMatrix(mat *Matrix) (*Matrix, error) {
	if m.rows != mat.rows || m.cols != mat.cols {
		return nil, errors.New("columns and rows must match")
	}
	// hadamard product
	return m.Map(func(val float64, i, j int) float64 {
		return val * mat.data[i][j]
	}), nil
}

//Subtract ...
func Subtract(a, b *Matrix) (*Matrix, error) {
	if a.rows != b.rows || a.cols != b.cols {
		return nil, errors.New("columns and rows of A and B must match")
	}

	return New(a.rows, a.cols).Map(func(val float64, i, j int) float64 {
		return a.data[i][j] - b.data[i][j]
	}), nil
}

//Add ...
func (m *Matrix) Add(n float64) *Matrix {
	return m.Map(func(val float64, _, _ int) float64 {
		return val + n
	})
}

//AddMatrix ...
func (m *Matrix) AddMatrix(mat *Matrix) (*Matrix, error) {
	if m.rows != mat.rows || m.cols != mat.cols {
		return nil, errors.New("columns and rows of matrices must match")
	}
	return m.Map(func(val float64, i, j int) float64 {
		return val + mat.data[i][j]
	}), nil
}

//Print ...
func (m *Matrix) Print() {
	fmt.Println(m.data)
}

//Serialize ...
func (m *Matrix) Serialize() ([]byte, error) {
	return json.Marshal(m)
}
