import re
import numpy as np


class InpParser:
    def __init__(self):
        self.elements = {}

    def _read_file(self, filename):
        with open(filename, "r", errors="ignore") as fp:
            return [line.rstrip("\n") for line in fp]

    def _split_csv_line(self, s):
        # ABAQUS lines are simple CSV-like, no quotes typically
        return [p.strip() for p in s.split(",") if p.strip()]

    def _find_kw(self, header, key):
        m = re.search(rf"\b{re.escape(key)}\s*=\s*([^,\s]+)", header, flags=re.I)
        return m.group(1) if m else None

    def parse_inp(self, filename):
        # Read the entire file in
        lines = self._read_file(filename)

        self.X = {}
        self.elem_conn = {}

        elem_type = None

        index = 0
        section = None
        while index < len(lines):
            raw = lines[index].strip()
            index += 1

            if not raw or raw.startswith("**"):
                continue

            if raw.startswith("*"):
                header = raw.upper()

                if header.startswith("*NODE"):
                    section = "NODE"
                elif header.startswith("*ELEMENT"):
                    section = "ELEMENT"
                    elem_type = self._find_kw(header, "TYPE")
                    elset = self._find_kw(header, "ELSET")
                    if elem_type not in self.elem_conn:
                        self.elem_conn[elset] = {}
                    if elset not in self.elem_conn[elset]:
                        self.elem_conn[elset][elem_type] = {}
                continue

            if section == "NODE":
                parts = self._split_csv_line(raw)
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                self.X[nid - 1] = (x, y, z)

            elif section == "ELEMENT":
                parts = self._split_csv_line(raw)
                eid = int(parts[0])
                conn = [(int(p) - 1) for p in parts[1:]]
                self.elem_conn[elset][elem_type][eid - 1] = conn

    def get_nodes(self):
        return np.array([self.X[k] for k in sorted(self.X.keys())])

    def get_domains(self):
        names = {}
        for elset in self.elem_conn:
            names[elset] = []
            for elem_type in self.elem_conn[elset]:
                names[elset].append(elem_type)

        return names

    def get_conn(self, elset, elem_type):
        conn = self.elem_conn[elset][elem_type]
        return np.array([conn[k] for k in sorted(conn.keys())], dtype=int)

    def get_edge_conn(self, elset, elem_type, edge_counter=0):
        """
        edge_counter is the offset to the edge tags. This is used when you have multiple regions in your mesh.
        """
        conn = self.get_conn(elset, elem_type)
        edge_node_conn = []  # edge -> node
        elem_edge_conn = []  # elem -> edge

        if conn.shape[1] != 3:
            raise ValueError("This only works for 3 node triangle")

        # edge_counter = 0
        for n0, n1, n2 in conn:
            # Define the edges
            n0 = int(n0)
            n1 = int(n1)
            n2 = int(n2)
            E0 = [n0, n1]
            E1 = [n1, n2]
            E2 = [n2, n0]

            E0_tag = edge_counter
            E1_tag = edge_counter + 1
            E2_tag = edge_counter + 2

            edge_node_conn.append(E0)
            edge_node_conn.append(E1)
            edge_node_conn.append(E2)

            elem_edge_conn.append([E0_tag, E1_tag, E2_tag])

            edge_counter += 3

        return edge_node_conn, elem_edge_conn
