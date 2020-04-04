from typing import Dict


class TransformationGraph():
    def __init__(self, dataset) -> None:

        self.roots = []

        built_nodes = {}

        def compute_transformation_graph(dataset):

            if dataset in built_nodes:
                return built_nodes[dataset]

            node: Dict = {
                "edge": None,
            }

            built_nodes[dataset] = node

            origin = dataset._get_origin()

            if type(origin) == list:
                node["edge"] = (list(map(
                    lambda partial_origin: {
                        **partial_origin,
                        "parent": compute_transformation_graph(
                            partial_origin["dataset"]
                        )
                    },
                    origin
                )))
            else:

                node["edge"] = {
                    **origin,
                    "parent": None,
                }

                if "dataset" in origin:
                    node["edge"]["parent"] = \
                        compute_transformation_graph(origin["dataset"])
                elif "root" in origin:

                    root = origin["root"]
                    if (root not in self.roots):
                        self.roots.append(root)

            return node

        self.graph = compute_transformation_graph(dataset)

    def display(self):
        def print_graph(node, delta=-1):
            def print_value(edge, delta):

                for i in range(delta - 1):
                    print("    ", end="")
                if delta > 0:
                    print("|---", end="")

                value = {}

                for key, val in edge.items():
                    if key not in ["parent", "dataset"]:
                        value[key] = val

                print(value)

            current_delta = delta

            while node is not None:
                current_delta += 1

                if type(node["edge"]) == list:
                    for edge in node["edge"]:
                        print_value(edge, current_delta)
                        print_graph(edge["parent"], current_delta)
                    return
                else:
                    print_value(node["edge"], current_delta)
                    node = node["edge"]["parent"]

        print_graph(self.graph)
