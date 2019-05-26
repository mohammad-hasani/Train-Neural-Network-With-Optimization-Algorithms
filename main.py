from Tools import get_data, get_weights, set_weights, build_model, evaluate_model


def main():
	X, y = get_data()
	model = build_model(X, y)
	weights, shape = get_weights(model)
	model = set_weights(model, weights, shape)
	score = evaluate_model(model, X, y)
	print(score)


if __name__ == '__main__':
	main()
