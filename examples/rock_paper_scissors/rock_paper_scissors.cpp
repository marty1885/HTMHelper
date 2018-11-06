#include "HTMHelper/HTMHelper.hpp"
using namespace HTM;

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>

#include <iostream>
#include <memory>

const int TP_DEPTH = 32;
const int ENCODE_WIDTH = 24;

enum Move
{
	Rock,
	Paper,
	Scissor
};

//Converts agent predictions to agent move
int predToMove(int pred)
{
	if(pred == Rock)
		return Paper;
	else if(pred == Paper)
		return Scissor;
	else
		return Rock;
}

//1 - first agent wins
//0 - draw
//-1 -second agent winds
int winner(int move1, int move2)
{
	if(move1 == move2)
		return 0;
	if(move1 == Rock && move2 == Paper)
		return -1;
	
	if(move1 == Paper && move2 == Scissor)
		return -1;
	
	if(move1 == Scissor && move2 == Rock)
		return -1;
	
	return 1;
}

std::string move2String(int move)
{
	if(move == Rock)
		return "Rock";
	else if(move == Paper)
		return "Paper";
	return "Scissor";
}

struct Player
{
	virtual xt::xarray<float> compute(int last_oppo_move, bool learn = true) = 0;
};

struct HTMPlayer : public Player
{
public:
	HTMPlayer() :
		tm({3*ENCODE_WIDTH}, TP_DEPTH), encoder(3, ENCODE_WIDTH)
	{
		tm->setMaxNewSynapseCount(64);
		tm->setPermanenceIncrement(0.1);
		tm->setPermanenceDecrement(0.045);
		tm->setConnectedPermanence(0.4);
		tm->setPredictedSegmentDecrement(0.3*2.0f*tm->getPermanenceIncrement());
	}
	
	virtual xt::xarray<float> compute(int last_oppo_move, bool learn = true) override
	{
		auto out = compute(encoder(last_oppo_move), true);
		return categroize(3, ENCODE_WIDTH, out);
	}
	
	xt::xarray<bool> train(const xt::xarray<bool>& x) {return compute(x, true);}
	xt::xarray<bool> predict(const xt::xarray<bool>& x) {return compute(x, false);}
	xt::xarray<bool> compute(const xt::xarray<bool>& x, bool learn) {return tm.compute(x, learn);}
	void reset() {tm.reset();}

	TemporalMemory tm;
	CategoryEncoder encoder;
};

struct HumanPlayer : public Player
{
	virtual xt::xarray<float> compute(int last_oppo_move, bool learn = true) override
	{
		int decition;
		do {
			std::cin >> decition;
		} while (decition < 1 || decition > 3);
		decition -= 1;

		xt::xarray<float> res = xt::zeros<float>({3});
		res[decition] = 1;
		return res;
	}
};

int main(int argc, char** argv)
{
	bool auto_play = false;
	if(argc > 1) {
		std::string opt = argv[1];
		if(opt == "-a" || opt == "--auto") {
			auto_play = true;

		}
		else {
			std::cout << "Usage: rock_paper_scissors [-h|--help] [-a|--auto]\n\t-h help\n\t-a auto play" << std::endl;
			return 0;
		}
	}

	std::cout << "Welcome to the Rock Paper Scissors game. In this game you (P1) is playing against an AI player implemnted in HTM."
	" This agent will learn your patterns as the game goes on and tetermin the next move according to the predictions it made."
	"\nYou can also use \033[1;37mrock_paper_scissors -a\033[0m to let the AI play against it self.\n"
	"Have fun!\n\n";

	//Initialize both AI
	std::unique_ptr<Player> player1;
	std::unique_ptr<Player> player2 = std::make_unique<HTMPlayer>();

	if(auto_play == true)
		player1 = std::make_unique<HTMPlayer>();
	else
		player1 = std::make_unique<HumanPlayer>();
	
	int p1_last_move = 0;
	int p2_last_move = 1;

	size_t p1_win = 0;
	size_t draw = 0;
	size_t p2_win = 0;

	int num_games = 1000;
	for(int i=0;i<num_games;i++) {
		std::cout << "Round " << i << '\n';
		if(auto_play == false)
			std::cout << "Please enter your move. 0 for Rock, 1 for Paper and 2 for Scissors.\n[1/2/3]: ";
		//Run Player1
		auto p1_out = player1->compute(p2_last_move);
		int p1_pred = xt::argmax(p1_out)[0];
		
		//Run Player2
		auto p2_out = player2->compute(p1_last_move);
		int p2_pred = xt::argmax(p2_out)[0];
		
		int p1_move = p1_pred;
		if(auto_play == true)
			 p1_move = predToMove(p1_pred);

		int p2_move = predToMove(p2_pred);
		
		int winner_player = winner(p1_move, p2_move);
		std::cout << "P1: " << move2String(p1_move) << ", " << "P2: " << move2String(p2_move)
			<< ", Winner: "<< (winner_player==1?"\033[0;32mP1\033[0m":(winner_player==0?"draw":"\033[0;31mP2\033[0m")) << '\n';

		
		p1_last_move = p1_move;
		p2_last_move = p2_move;
		
		if(winner_player == 1)
			p1_win += 1;
		else if(winner_player == 0)
			draw += 1;
		else
			p2_win += 1;

		std::cout << "P1 win rate: " << (float)p1_win/(p1_win+p2_win) << "P2 win rate: " << (float)p2_win/(p1_win+p2_win) << '\n';
		std::cout << std::endl;
	}

	std::cout << "After all the battles" << std::endl;
	std::cout << "P1 Wins " << p1_win << " times, " << (float)p1_win/num_games << "\n";
	std::cout << "P2 Wins " << p2_win << " times, " << (float)p2_win/num_games << "\n";
	std::cout << "draw: " << draw << std::endl;


}