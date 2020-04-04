from typing import Dict
import copy
import dill
import base64


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

    def serialize(self) -> str:
        def minimize(node):

            current_node = node

            def minimized_value(edge):

                value = {}

                for key, val in edge.items():
                    if key not in ["dataset"]:
                        value[key] = val

                return value

            while current_node is not None:
                if type(current_node["edge"]) == list:
                    for i, edge in enumerate(current_node["edge"]):
                        current_node["edge"][i] = minimized_value(edge)
                        minimize(edge["parent"])
                    return
                else:
                    current_node["edge"] = minimized_value(
                        current_node["edge"]
                    )
                    current_node = current_node["edge"]["parent"]

        result = copy.deepcopy(self.graph)
        minimize(result)

        result = base64.b64encode(dill.dumps(result)).decode()

        return result

    def is_same_as_serialized(self, serialized):
        return self.serialize() == serialized
