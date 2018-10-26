function docNode = struct2DOMnode(s)

docNode = com.mathworks.xml.XMLUtils.createDocument(s.Name);
node = docNode.getDocumentElement;
[node, docNode] = createNode(docNode, node, s.Children);

function [update_node, update_docNode] = createNode(docNode, node, s)
update_node = node;
update_docNode = docNode;
for n = 1:length(s)
    s_name = s(n).Name;
    s_data = s(n).Data;
    s_children = s(n).Children;
    check_name = isstrprop(s_name,'alphanum');
    check_data = isstrprop(s_data,'alphanum');
    
    if check_name(1)==0 && (isempty(check_data) || (check_data(1)==0 && s_data(1)~='/'))
        continue
    end
    
    if check_name(1)==1
        child_node = update_docNode.createElement(s_name);
        [parent_node,update_docNode] = createNode(update_docNode, child_node, s_children);
        update_node.appendChild(parent_node);
    else
        child_node = update_docNode.createTextNode(s_data);
        update_node.appendChild(child_node);
    end
end