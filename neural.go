package main

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/emicklei/dot"
)

var (
	SigmoidConst = 100.0
)

type Sample struct {
	X []uint8
	Y []uint8
}

type ActivationFunction func(float64) float64

// Activation Functions
func Identity(v float64) float64 {
	return v
}

func Threshold(v float64) float64 {
	if v >= 0 {
		return 1
	} else {
		return 0
	}
}

func PiecewiseLinear(v float64) float64 {
	if v >= 0.5 {
		return 1
	} else if v > -0.5 && v < 0.5 {
		return v
	} else {
		return 0
	}
}

func Sigmoid(v float64) float64 {
	return 1. / (1. + math.Exp(-SigmoidConst*v))
}

type Edge struct {
	Source     *Node
	Target     *Node
	Weight     float64
	NewWeight  float64
	PrevWeight float64 // used for momentum
}

func NewEdge(source, target *Node) *Edge {
	return &Edge{
		Source: source,
		Target: target,
		Weight: randomWeight(),
	}
}

func randomWeight() float64 {
	desiredStdDev := 0.01
	desiredMean := 0.0
	return rand.NormFloat64()*desiredStdDev + desiredMean
}

type Node struct {
	ID         int
	Activation ActivationFunction
	In         []*Edge
	Out        []*Edge

	inducedLocalField float64
	delta             float64
	permanentMark     bool
	temporaryMark     bool

	isInput  bool
	isOutput bool
}

func (n *Node) outputSignal() float64 {
	if n.isInput {
		return n.inducedLocalField
	}
	return n.Activation(n.inducedLocalField)
}

func (n *Node) ConnectToSource(src *Node) {
	edge := NewEdge(src, n)
	n.In = append(n.In, edge)
	src.Out = append(src.Out, edge)
}

func (n *Node) ConnectToTarget(tgt *Node) {
	edge := NewEdge(n, tgt)
	n.Out = append(n.Out, edge)
	tgt.In = append(tgt.In, edge)
}

type Neural struct {
	nodes    []*Node
	forward  []*Node // topologically sorted from source to target
	backward []*Node // topologically sorted from target to source

	inputs  []*Node
	outputs []*Node

	bias     *Node
	analyzed bool

	nextID int

	LearningRate float64
	Momentum     float64
}

func NewNeural(inputs, outputs int) *Neural {
	n := &Neural{
		nodes:        make([]*Node, 0, 10),
		LearningRate: 0.01,
		Momentum:     0.0,
		analyzed:     false,
	}

	n.CreateInputNodes(inputs, Sigmoid)
	n.CreateOutputNodes(outputs, Sigmoid)

	return n
}

// Analyze applies topological sort in both directions. This assumes
// that the network is a directed acyclic graph. It will panic if it is not.

// L ← Empty list that will contain the sorted nodes
// while exists nodes without a permanent mark do
//   select an unmarked node n
// 	 visit(n)
// function visit(node n)
// 	 if n has a permanent mark then return
// 	 if n has a temporary mark then stop   (not a DAG)
// 	 mark n with a temporary mark
// 	 for each node m with an edge from n to m do
// 	   visit(m)
// 	 remove temporary mark from n
// 	 mark n with a permanent mark
// 	 add n to head of L
func (n *Neural) Analyze() {
	n.forward = make([]*Node, 0, len(n.nodes))
	n.backward = make([]*Node, 0, len(n.nodes))

	reset := func() {
		for _, node := range n.nodes {
			node.temporaryMark = false
			node.permanentMark = false
		}
	}

	// forward topological sort
	reset()

	// visit unmarked nodes
	for _, node := range n.nodes {
		if node.permanentMark {
			continue
		}
		n.dfsVisit(node, true)
	}

	// backward topological sort
	reset()

	// visit unmarked nodes
	for _, node := range n.nodes {
		if node.permanentMark {
			continue
		}
		n.dfsVisit(node, false)
	}

	n.analyzed = true
}

func (n *Neural) dfsVisit(node *Node, forward bool) {
	if node.permanentMark {
		return
	}
	if node.temporaryMark {
		panic("Not a DAG")
	}
	node.temporaryMark = true
	if forward {
		for _, e := range node.In {
			n.dfsVisit(e.Source, forward)
		}
	} else {
		for _, e := range node.Out {
			n.dfsVisit(e.Target, forward)
		}
	}
	node.temporaryMark = false
	node.permanentMark = true

	if forward {
		n.forward = append(n.forward, node)
	} else {
		n.backward = append(n.backward, node)
	}
}

func (n *Neural) Inputs() []*Node {
	return n.inputs
}

func (n *Neural) Outputs() []*Node {
	return n.outputs
}

func (n *Neural) CreateInputNodes(size int, a ActivationFunction) {
	for i := 0; i < size; i++ {
		node := n.AddNode(a)
		node.isInput = true
		n.inputs = append(n.inputs, node)
	}
}

func (n *Neural) CreateOutputNodes(size int, a ActivationFunction) {
	for i := 0; i < size; i++ {
		node := n.AddNode(a)
		node.isOutput = true
		n.outputs = append(n.outputs, node)
	}
}

func (n *Neural) AddNode(a ActivationFunction) *Node {
	node := &Node{
		ID:         n.nextID,
		Activation: a,
	}
	n.nextID++
	n.nodes = append(n.nodes, node)
	n.analyzed = false
	return node
}

func (n *Neural) AddBias() {
	n.bias = &Node{
		ID:                -1,
		Activation:        Identity,
		inducedLocalField: 1,
	}
	for i := range n.nodes {
		if !n.nodes[i].isInput {
			n.nodes[i].ConnectToSource(n.bias)
		}
	}

	n.nodes = append(n.nodes, n.bias)
	n.analyzed = false
}

func (n *Neural) Print() {
	if !n.analyzed {
		n.Analyze()
	}

	g := dot.NewGraph(dot.Directed)
	g.Attr("rankdir", "LR")

	queue := make([]*Node, 0)
	for i := range n.inputs {
		queue = append(queue, n.inputs[i])
	}

	gNodes := map[int]dot.Node{}
	for _, x := range n.forward {
		gn := g.Node(fmt.Sprintf("%d", x.ID))
		if _, ok := gNodes[x.ID]; !ok {
			gNodes[x.ID] = gn
		}
		for _, edge := range x.In {
			g.Edge(gNodes[edge.Source.ID], gn).Label(fmt.Sprintf("%.2f", edge.Weight))
		}
	}

	fmt.Println(g.String())
}

func (n *Neural) Train(X, Y [][]uint8) error {
	if len(X) != len(Y) {
		return errors.New("len(X) != len(Y)")
	}
	if !n.analyzed {
		n.Analyze()
	}

	for epoch := 0; epoch < 1000; epoch++ {
		e := make([]float64, len(n.outputs))

		for i := range rand.Perm(len(X)) {
			d := Y[i]
			o := n.Predict(X[i])

			for j := range e {
				e[j] = float64(d[j]) - o[j]
			}

			// adjust weights
			n.AdjustWeights(e)
		}

		// print error after epoch
		// log.Printf("Epoch %d: error %.3f", epoch, n.AverageError(X, Y))
	}
	return nil
}

func (n *Neural) AdjustWeights(errorSignal []float64) {
	if len(errorSignal) != len(n.outputs) {
		panic("len(errorSignal) != len(n.outputs)")
	}
	for _, node := range n.nodes {
		node.delta = 0
	}

	for i, node := range n.outputs {
		node.delta = errorSignal[i] * n.outputs[i].outputSignal()
	}

	for _, node := range n.backward {
		for _, edge := range node.In {
			edge.Source.delta += node.delta * edge.Weight
		}
		if node.isOutput {
			continue
		}

		node.delta = SigmoidConst * node.outputSignal() * (1 - node.outputSignal()) * node.delta

		for _, edge := range node.Out {
			edge.NewWeight = edge.Weight + n.Momentum*edge.PrevWeight + n.LearningRate*edge.Target.delta*edge.Source.outputSignal()
		}
	}

	for _, node := range n.nodes {
		for _, edge := range node.Out {
			edge.PrevWeight = edge.Weight
			edge.Weight = edge.NewWeight
			edge.NewWeight = 0
		}
	}
}

func (n *Neural) TrainOnSamples(samples []Sample) error {
	X := make([][]uint8, len(samples))
	Y := make([][]uint8, len(samples))
	for i := range samples {
		X[i] = samples[i].X
		Y[i] = samples[i].Y
	}
	return n.Train(X, Y)
}

func (n *Neural) Predict(X []uint8) []float64 {
	if len(X) != len(n.inputs) {
		panic(fmt.Sprintf("len(X) = %d, should be %d", len(X), len(n.inputs)))
	}
	if !n.analyzed {
		n.Analyze()
	}

	// reset nodes
	for _, node := range n.nodes {
		if node == n.bias {
			continue
		}
		node.inducedLocalField = 0
	}

	for i := range n.inputs {
		n.inputs[i].inducedLocalField = float64(X[i])
	}

	for _, node := range n.forward {
		for _, edge := range node.Out {
			edge.Target.inducedLocalField += edge.Weight * node.outputSignal()
		}
	}

	Y := make([]float64, len(n.outputs))
	for i := range n.outputs {
		Y[i] = n.outputs[i].outputSignal()
	}
	return Y
}

func (n *Neural) Error(X, Y []uint8) float64 {
	got := n.Predict(X)
	e := 0.0
	for i := range got {
		e += math.Pow(got[i]-float64(Y[i]), 2.0)
	}
	return math.Sqrt(e)
}

func (n *Neural) AverageError(X, Y [][]uint8) float64 {
	e := 0.0
	for i := range X {
		e += n.Error(X[i], Y[i])
	}
	e /= float64(len(X))
	return e
}

// Draw example network for XOR
func ExampleXOR() {
	n := NewNeural(2, 1)
	hiddenNodes := make([]*Node, 0, 2)
	for i := 0; i < 2; i++ {
		node := n.AddNode(Sigmoid)
		for _, inp := range n.Inputs() {
			inp.ConnectToTarget(node)
		}
		for _, out := range n.Outputs() {
			node.ConnectToTarget(out)
		}
		hiddenNodes = append(hiddenNodes, node)
	}
	hiddenNodes[0].Out[0].Weight = -2
	hiddenNodes[1].Out[0].Weight = 1
	n.AddBias()

	n.bias.Out[0].Weight = -0.5 // output
	n.bias.Out[1].Weight = -1.5 // hidden node
	n.bias.Out[2].Weight = -0.5 // hidden node

	for _, inp := range n.inputs {
		for _, edge := range inp.Out {
			edge.Weight = 1.0
		}
	}

	n.Print()
}

func main() {
	n := NewNeural(2, 1)
	for i := 0; i < 10; i++ {
		node := n.AddNode(Sigmoid)
		for _, inp := range n.Inputs() {
			inp.ConnectToTarget(node)
		}
		for _, out := range n.Outputs() {
			node.ConnectToTarget(out)
		}
	}

	n.AddBias()

	// test it out
	samples := []Sample{
		{X: []uint8{0, 0}, Y: []uint8{0}},
		{X: []uint8{0, 1}, Y: []uint8{1}},
		{X: []uint8{1, 0}, Y: []uint8{1}},
		{X: []uint8{1, 1}, Y: []uint8{0}},
	}
	log.Println("Before training")
	for i := range samples {
		log.Println(samples[i].X, n.Predict(samples[i].X))
	}
	n.TrainOnSamples(samples)

	log.Println("After training")
	for i := range samples {
		log.Println(samples[i].X, n.Predict(samples[i].X))
	}
	n.Print()
}
