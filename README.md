# Neural

A neural network implementation in Go. This is still under development. The ultimate aim is to have a usable library that can also do weight-agnostic training.

## Usage

See neural_test.go for examples. I will add real usage instructions here once the library is more mature.

## Draw Network Diagram

Draw the network diagram for debugging:

```
go run neural.go | dot -Tpng  > test.png && open test.png
```