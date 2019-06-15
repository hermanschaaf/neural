package main

import (
	"math"
	"testing"
)

func absDiff(a, b float64) float64 {
	return math.Abs(a - b)
}

func TestNeural_Predict(t *testing.T) {
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

	got := n.Predict([]uint8{0, 0})
	if absDiff(got[0], 0.0) > 0.01 {
		t.Errorf("got n.Predict([0,0]) = %.2f, want %.2f", got[0], 0.0)
	}

	got = n.Predict([]uint8{1, 0})
	if absDiff(got[0], 1.0) > 0.01 {
		t.Errorf("got n.Predict([1,0]) = %.2f, want %.2f", got[0], 1.0)
	}
}

func TestNeural_TrainOnSamples_XOR(t *testing.T) {
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
	n.TrainOnSamples(samples)
	for i := range samples {
		give := samples[i].X
		got := n.Predict(give)
		want := float64(samples[i].Y[0])
		if absDiff(got[0], want) > 0.05 {
			t.Errorf("XOR Problem: Predict(%v) = %v, want %v", give, got, want)
		}
	}
}
