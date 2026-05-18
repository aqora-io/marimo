import { type ReactElement, type ReactNode, isValidElement } from "react";

export function isReactNode(node: unknown): node is ReactNode {
  return (
    node === null ||
    node === undefined ||
    typeof node === "string" ||
    typeof node === "number" ||
    typeof node === "boolean" ||
    isValidElement(node) ||
    (Array.isArray(node) && node.every(isReactNode))
  );
}

export function isComplexReactNode(
  node: unknown,
): node is ReactElement | Array<ReactNode> {
  return (
    isValidElement(node) || (Array.isArray(node) && node.every(isReactNode))
  );
}
