package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"os"
	"strings"
	"time"

	"gocv.io/x/gocv"
)

// getOutputsNames : YOLO Layer
func getOutputsNames(net *gocv.Net) []string {
	var outputLayers []string
	for _, i := range net.GetUnconnectedOutLayers() {
		layer := net.GetLayer(i)
		layerName := layer.GetName()
		if layerName != "_input" {
			outputLayers = append(outputLayers, layerName)
		}
	}
	return outputLayers
}

// PostProcess : All Detect Box
func PostProcess(frame gocv.Mat, outs *[]gocv.Mat) ([]image.Rectangle, []float32, []int) {
	var classIds []int
	var confidences []float32
	var boxes []image.Rectangle
	for _, out := range *outs {

		data, _ := out.DataPtrFloat32()
		for i := 0; i < out.Rows(); i, data = i+1, data[out.Cols():] {

			scoresCol := out.RowRange(i, i+1)

			scores := scoresCol.ColRange(5, out.Cols())
			_, confidence, _, classIDPoint := gocv.MinMaxLoc(scores)
			//if confidence > 0.5 {

			centerX := int(data[0] * float32(frame.Cols()))
			centerY := int(data[1] * float32(frame.Rows()))
			width := int(data[2] * float32(frame.Cols()))
			height := int(data[3] * float32(frame.Rows()))

			left := centerX - width/2
			top := centerY - height/2
			classIds = append(classIds, classIDPoint.X)
			confidences = append(confidences, float32(confidence))
			boxes = append(boxes, image.Rect(left, top, width, height))
			//}
		}
	}
	return boxes, confidences, classIds
}

// ReadCOCO : Read coco.names
func ReadCOCO() []string {
	var classes []string
	read, _ := os.ReadFile("./assets/coco.names")
	classes = strings.Split(string(read), "\n")
	return classes
}

// drawRect : Detect Class to Draw Rect
func drawRect(img gocv.Mat, boxes []image.Rectangle, classes []string, confidences []float32, classIds []int, indices []int) (gocv.Mat, []string, []float32) {
	var detectClass []string
	var detectConfidence []float32
	for _, idx := range indices {
		if idx == 0 {
			continue
		}
		gocv.Rectangle(&img, image.Rect(boxes[idx].Max.X, boxes[idx].Max.Y, boxes[idx].Max.X+boxes[idx].Min.X, boxes[idx].Max.Y+boxes[idx].Min.Y), color.RGBA{255, 0, 0, 0}, 2)
		gocv.PutText(&img, classes[classIds[idx]], image.Point{boxes[idx].Max.X, boxes[idx].Max.Y + 30}, gocv.FontHersheyDuplex, 1, color.RGBA{0, 0, 255, 0}, 3)
		detectClass = append(detectClass, classes[classIds[idx]])
		detectConfidence = append(detectConfidence, confidences[idx])
	}
	return img, detectClass, detectConfidence
}

// Detect : Run YOLOv4 Process
func Detect(net *gocv.Net, src gocv.Mat, networkSize int, scoreThreshold float32, nmsThreshold float32, OutputNames []string, classes []string) (gocv.Mat, []string, []float32) {
	img := src.Clone()
	img.ConvertTo(&img, gocv.MatTypeCV32F)
	blob := gocv.BlobFromImage(img, 1/255.0, image.Pt(networkSize, networkSize), gocv.NewScalar(0, 0, 0, 0), true, false)
	net.SetInput(blob, "")
	probs := net.ForwardLayers(OutputNames)
	boxes, confidences, classIds := PostProcess(img, &probs)

	indices := make([]int, 100)
	if len(boxes) == 0 { // No Classes
		return src, []string{}, []float32{}
	}
	gocv.NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, indices)

	return drawRect(src, boxes, classes, confidences, classIds, indices)
}

// Process : Read Picture and Process
func Process(networkName string, networkSize int, threshold float64, nmsThreshold float64, filename string) {
	// Init
	classes := ReadCOCO()

	net := gocv.ReadNet("./assets/"+networkName+".weights", "./assets/"+networkName+".cfg")
	//net := gocv.ReadNet("./assets/yolov4-tiny.weights", "./assets/yolov4-tiny.cfg")
	//net := gocv.ReadNet("./assets/enet-coco.weights", "./assets/enet-coco.cfg")
	//net := gocv.ReadNet("../darknet/yolov4-p6.weights", "../darknet/cfg/yolov4-p6.cfg")
	defer net.Close()
	//net.SetPreferableBackend(gocv.NetBackendCUDA)
	//net.SetPreferableTarget(gocv.NetTargetCUDA)

	OutputNames := getOutputsNames(&net)

	//window := gocv.NewWindow("yolo")
	img := gocv.IMRead(filename, gocv.IMReadColor)
	defer img.Close()

	startTime := time.Now()
	detectImg, detectClass, detectConfidence := Detect(&net, img.Clone(), networkSize, float32(threshold), float32(nmsThreshold), OutputNames, classes)
	endTime := time.Now()
	defer detectImg.Close()

	fmt.Println("Time taken: ", endTime.Sub(startTime))
	fmt.Printf("Dectect Class : %v\n", detectClass)
	fmt.Printf("Dectect Confidence : %v\n", detectConfidence)
	//window.IMShow(detectImg)
	gocv.IMWrite("result.jpg", detectImg)
	gocv.WaitKey(0)

}

func main() {
	networkName := flag.String("net", "", "Neural network name")
	networkSize := flag.Int("size", 416, "Neural network size")
	threshold := flag.Float64("thresh", 0.45, "Threshold")
	nmsThreshold := flag.Float64("nms", 0.8, "NMS Threshold")
	imageFilename := flag.String("image", "", "Image input filename")
	flag.Parse()
	Process(*networkName, *networkSize, *threshold, *nmsThreshold, *imageFilename)
}
