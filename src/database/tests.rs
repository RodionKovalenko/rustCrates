#[cfg(test)]
mod tests {
    use crate::database::crud_service::{get_value, insert_token};

    #[test]
    fn test_save() {
        let _ = insert_token("A");
        let value = get_value("A").unwrap();
        println!("value is {}", &value);
    
        let _ = insert_token("B");
        let value = get_value("B").unwrap();
        println!("value is {}", &value);
    
        let _ = insert_token("C");
        let value = get_value("C").unwrap();
        println!("value is {}", &value);
    
        let _ = insert_token("D");
        let value = get_value("D").unwrap();
        println!("value is {}", &value);
    }
}