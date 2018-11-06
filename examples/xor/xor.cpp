#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>

#include "HTMHelper/HTMHelper.hpp"

using namespace HTM;

const int CATEGORY_LEN = 16;
const int TP_DEPTH = 16;
const int NUM_EPOCH = 4;

//Abusing the TemporalPooler to solve the XOR problem
int main()
{
	TemporalPooler tp({CATEGORY_LEN*2}, TP_DEPTH);
	CategoryEncoder encoder(2, CATEGORY_LEN);

	xt::xarray<int> x({{0,0}, {0,1}, {1,0}, {1,1}});
	xt::xarray<int> y({0, 1, 1, 0});
	
	//Train the TP
	for(int i=0;i<NUM_EPOCH;i++) {
		for(int j=0;j<4;j++) {
			auto input = xt::view(x, j);
			auto output = xt::view(y, j);
			
			//Train the TemporalPooler with the sequence
			//First the two inputs
			tp.train(encoder(input[0]));
			tp.train(encoder(input[1]));

			//Then the desired output
			tp.train(encoder(output[0]));

			//The sequence has finished. Reset the sate to learn the next sequence
			tp.reset();
		}
	}

	for(int j=0;j<4;j++) {
		auto input = xt::view(x, j);
		auto output = xt::view(y, j);
		
		//Run trough the sequence
		tp.predict(encoder(input[0]));
		tp.predict(encoder(input[1]));

		//Get the predicted sequence
		auto pred = tp.predict(encoder(output[0]));

		//Convert the prediction into predicted categories
		auto prop = categroize(2, CATEGORY_LEN, pred);
		auto prediction = xt::argmax(prop)[0];

		std::cout << input[0] << " xor " << input[1]
			<< " = "<< prediction << std::endl;

		tp.reset();
	}
}